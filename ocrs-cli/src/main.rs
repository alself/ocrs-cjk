use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::io::{BufWriter, IsTerminal, Read};

use anyhow::{anyhow, Context};
use ocrs_cjk::{
    DecodeMethod, DimOrder, ImageSource, OcrEngine, OcrEngineParams, OcrInput, TextDetectorParams,
    TextItem,
};
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

mod deskew;
mod doctor;
mod furigana;
mod models;
mod post_correct;
use models::{load_model, ModelSource};
mod output;
#[cfg(not(target_arch = "wasm32"))]
mod pdf;
mod table;
use output::{
    format_alto_output, format_alto_pdf_output, format_blocks_json_output,
    format_furigana_json_output, format_hocr_output, format_hocr_pdf_output, format_json_output,
    format_json_pdf_output, format_markdown_output, format_markdown_pdf_output,
    format_review_json_output, format_text_output, format_tsv_output, format_tsv_pdf_output,
    generate_annotated_png, generate_confidence_heatmap, FormatJsonArgs, GeneratePngArgs,
    OutputFormat, PageInfo,
};

/// Write a CHW image to a PNG file in `path`.
fn write_image(path: &str, img: NdTensorView<f32, 3>) -> anyhow::Result<()> {
    let img_width = img.size(2);
    let img_height = img.size(1);
    let color_type = match img.size(0) {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        chans => return Err(anyhow!("Unsupported channel count {}", chans)),
    };

    let hwc_img = img.permuted([1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}

/// Convert an CHW float tensor with values in the range [0, 1] to `Vec<u8>`
/// with values scaled to [0, 255].
fn image_from_tensor(tensor: NdTensorView<f32, 3>) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

/// Source of the input image.
enum InputSource {
    /// Read from a file path.
    File(String),
    /// Read from stdin.
    Stdin,
    /// Read from the system clipboard.
    Clipboard,
}

/// Extract images of individual text lines from `img`, apply the same
/// preprocessing that would be applied before text recognition, and save
/// in PNG format to `output_dir`.
fn write_preprocessed_text_line_images(
    input: &OcrInput,
    engine: &OcrEngine,
    line_rects: &[Vec<RotatedRect>],
    output_dir: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create dir {}/", output_dir))?;

    for (line_index, word_rects) in line_rects.iter().enumerate() {
        let filename = format!("{}/line-{}.png", output_dir, line_index);
        let mut line_img = engine.prepare_recognition_input(input, word_rects.as_slice())?;
        line_img.apply(|x| x + 0.5);
        let shape = [1, line_img.size(0), line_img.size(1)];
        let line_img = line_img.into_shape(shape);
        write_image(&filename, line_img.view())
            .with_context(|| format!("Failed to write line image to {}", filename))?;
    }

    Ok(())
}

struct Args {
    /// Path to text detection model.
    detection_model: Option<String>,

    /// Path to text recognition model.
    recognition_model: Option<String>,

    /// Source of the input image.
    input: InputSource,

    /// Enable debug output.
    debug: bool,

    /// Detect and correct image skew before OCR.
    deskew: bool,

    output_format: OutputFormat,

    /// Output file path. Defaults to stdout.
    output_path: Option<String>,

    /// Use beam search for sequence decoding.
    beam_search: bool,

    /// Generate a text probability map.
    text_map: bool,

    /// Generate a text mask. This is the binarized version of the probability map.
    text_mask: bool,

    /// Extract each text line found and save as a PNG image.
    text_line_images: bool,

    /// Recognize each detected box independently, without merging boxes into
    /// lines first. Matches the PaddleOCR / RapidOCR pipeline.
    rec_per_box: bool,

    /// Expand each detected box outwards using PaddleOCR's `DBPostProcess`
    /// unclip ratio (`area * ratio / perimeter`). Only meaningful for
    /// PaddleOCR DB detection models.
    det_unclip: Option<f32>,

    /// Filter characters produced by text recognition
    /// This must be a sub-set of `alphabet`.
    allowed_chars: Option<String>,

    /// Alphabet used by the recognition model.
    /// If not provided, the default alphabet is used.
    alphabet: Option<String>,

    /// Path to a file containing the alphabet (all characters concatenated, no separator).
    /// Useful for large CJK alphabets where passing characters via --alphabet is impractical.
    alphabet_file: Option<String>,

    #[allow(dead_code)]
    model_dir: Option<String>,

    /// Write a searchable PDF (with invisible text overlay) to this path.
    output_pdf: Option<String>,

    /// Minimum confidence [0.0, 1.0] for words included in the PDF text layer.
    /// Words below this threshold are omitted from the invisible text overlay.
    min_confidence: Option<f32>,

    /// Confidence threshold for marking low-confidence words in output.
    low_confidence_mark: Option<f32>,

    /// Auto-detect and correct 90°/180°/270° image rotation.
    auto_rotate: bool,

    /// Apply Japanese OCR confusion corrections for chars below this confidence.
    post_correct_ja: Option<f32>,

    /// Convert full-width ASCII variants to half-width.
    normalize_ja: bool,

    /// Filter OCR output to lines intersecting [left, top, width, height].
    region: Option<[i32; 4]>,

    /// Filter OCR output to lines containing this text.
    find_text: Option<String>,

    /// Detect table-like text layouts and output as CSV/Markdown/JSON tables.
    tables: bool,

    /// Save a confidence heatmap PNG (red=low, green=high) to this path.
    confidence_heatmap: Option<String>,

    /// Skip OCR for PDF pages that have an existing text layer.
    skip_text_pages: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut allowed_chars = None;
    let mut alphabet = None;
    let mut alphabet_file = None;
    let mut beam_search = false;
    let mut clipboard = false;
    let mut debug = false;
    let mut deskew = false;
    let mut detection_model = None;
    let mut low_confidence_mark = None;
    let mut min_confidence = None;
    let mut find_text: Option<String> = None;
    let mut post_correct_ja = None;
    let mut normalize_ja = false;
    let mut region: Option<[i32; 4]> = None;
    let mut model_dir = None;
    let mut output_format = OutputFormat::Text;
    let mut output_path = None;
    let mut output_pdf = None;
    let mut recognition_model = None;
    let mut text_line_images = false;
    let mut text_map = false;
    let mut text_mask = false;
    let mut rec_per_box = false;
    let mut det_unclip = None;
    let mut lang = false;
    let mut tables = false;
    let mut confidence_heatmap: Option<String> = None;
    let mut auto_rotate = false;
    let mut skip_text_pages = false;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("allowed-chars") => {
                allowed_chars = Some(parser.value()?.string()?);
            }
            Short('a') | Long("alphabet") => {
                alphabet = Some(parser.value()?.string()?);
            }
            Long("alphabet-file") => {
                alphabet_file = Some(parser.value()?.string()?);
            }
            Long("beam") => {
                beam_search = true;
            }
            Short('c') | Long("clipboard") => {
                clipboard = true;
            }
            Long("debug") => {
                debug = true;
            }
            Long("deskew") => {
                deskew = true;
            }
            Long("auto-rotate") => {
                auto_rotate = true;
            }
            Long("detect-model") => {
                detection_model = Some(parser.value()?.string()?);
            }
            Long("min-confidence") => {
                let v: f32 = parser
                    .value()?
                    .string()?
                    .parse()
                    .map_err(|_| "invalid --min-confidence value (expected 0.0–1.0)")?;
                min_confidence = Some(v.clamp(0.0, 1.0));
            }
            Long("find-text") => {
                find_text = Some(parser.value()?.string()?);
            }
            Long("region") => {
                let s = parser.value()?.string()?;
                let parts: Vec<i32> = s
                    .split(',')
                    .map(|v| v.trim().parse::<i32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid --region: expected x,y,w,h (pixel integers)")?;
                if parts.len() != 4 {
                    return Err("--region requires exactly 4 values: x,y,w,h".into());
                }
                region = Some([parts[0], parts[1], parts[2], parts[3]]);
            }
            Long("post-correct-ja") => {
                let v: f32 = parser
                    .value()?
                    .string()?
                    .parse()
                    .map_err(|_| "invalid --post-correct-ja value (expected 0.0–1.0)")?;
                post_correct_ja = Some(v.clamp(0.0, 1.0));
            }
            Long("normalize-ja") => {
                normalize_ja = true;
            }
            Long("mark-low-confidence") => {
                let v: f32 = parser
                    .value()?
                    .string()?
                    .parse()
                    .map_err(|_| "invalid --mark-low-confidence value (expected 0.0–1.0)")?;
                low_confidence_mark = Some(v.clamp(0.0, 1.0));
            }
            Long("lang") => {
                let _v = parser.value()?.string()?; // ja/zh/ko/zh-tw accepted; all use same PP-OCRv5 model
                lang = true;
            }
            Long("model-dir") => {
                model_dir = Some(parser.value()?.string()?);
            }
            Long("output-pdf") => {
                output_pdf = Some(parser.value()?.string()?);
            }
            Long("confidence-heatmap") => {
                confidence_heatmap = Some(parser.value()?.string()?);
            }
            Long("skip-text-pages") => {
                skip_text_pages = true;
            }
            Long("alto") => {
                output_format = OutputFormat::Alto;
            }
            Long("hocr") => {
                output_format = OutputFormat::Hocr;
            }
            Short('j') | Long("json") => {
                output_format = OutputFormat::Json;
            }
            Short('m') | Long("markdown") => {
                output_format = OutputFormat::Markdown;
            }
            Long("tsv") => {
                output_format = OutputFormat::Tsv;
            }
            Long("blocks-json") => {
                output_format = OutputFormat::BlocksJson;
            }
            Long("furigana-separate") => {
                output_format = OutputFormat::FuriganaJson;
            }
            Long("review-json") => {
                output_format = OutputFormat::ReviewJson;
            }
            Long("tables") => {
                tables = true;
            }
            Short('o') | Long("output") => {
                output_path = Some(parser.value()?.string()?);
            }
            Short('p') | Long("png") => {
                output_format = OutputFormat::Png;
            }
            Long("rec-model") => {
                recognition_model = Some(parser.value()?.string()?);
            }
            Long("text-line-images") => {
                text_line_images = true;
            }
            Long("text-map") => {
                text_map = true;
            }
            Long("text-mask") => {
                text_mask = true;
            }
            Long("rec-per-box") => {
                rec_per_box = true;
            }
            Long("det-unclip") => {
                let v: f32 = parser
                    .value()?
                    .string()?
                    .parse()
                    .map_err(|_| "invalid --det-unclip value (expected a ratio, eg. 1.5)")?;
                det_unclip = Some(v.max(0.));
            }
            Long("help") => {
                println!(
                    "Extract text from an image.

Usage: {bin_name} [OPTIONS] [image]

  If no image path is given, reads from stdin.

Options:

  --allowed-chars <chars>

    Filter characters produced by text recognition

  -a, --alphabet <chars>

    Specify the alphabet used by the recognition model

  --alphabet-file <path>

    Read the alphabet from a file (characters concatenated, UTF-8).
    Use this instead of --alphabet for large CJK alphabets.

  -c, --clipboard

    Read image from system clipboard

  --auto-rotate

    Detect and correct 90°/180°/270° image rotation before OCR.
    Uses horizontal projection profile analysis to detect orientation.
    Apply before --deskew for best results. Use --debug to see correction angle.

  --deskew

    Detect and correct image skew before OCR (±15°, projection profile).
    Use --debug to print the detected skew angle.

  --detect-model <path>

    Use a custom text detection model

  --find-text <query>

    Filter OCR output to lines containing this text (case-sensitive, Unicode).
    Combine with --json to get bounding boxes of matching lines.

  --lang <ja|zh|ko|zh-tw>

    Use CJK-trained PP-OCRv5 models. Downloads detection and recognition
    models on first use (~85 MB each, cached in ~/.cache/ocrs/).
    Explicit --detect-model, --rec-model, or --alphabet flags take precedence.

  --blocks-json

    Output text as a blocks JSON document with reading order numbers and
    block-level bounding boxes. Each block has: type, reading_order, bbox,
    text, and lines. Blocks are paragraphs detected by vertical gap analysis.
    PDF input: not supported.

  --furigana-separate

    Detect and separate furigana (ルビ) from main text. Outputs JSON with
    \"text\" (main kanji/text) and \"ruby\" (associated kana annotation) per line.
    Furigana lines are detected by character height ratio and kana proportion.
    PDF input: not supported.

  -j, --json

    Output text and structure in JSON format

  --review-json

    Output JSON with per-word confidence, needs_review flags, and suspected_chars
    for words below the threshold. Combine with --mark-low-confidence to set a
    custom threshold (default 0.7). Useful for human-in-the-loop OCR workflows.
    PDF input: not supported.

  -m, --markdown

    Output extracted text in Markdown format (each OCR line as a paragraph).
    PDF pages are separated by horizontal rules (---).

  --mark-low-confidence <0.0–1.0>

    Mark words with confidence below this threshold.

  --post-correct-ja <0.0–1.0>

    Apply Japanese OCR confusion corrections (0/O, 1/l, 曰→日) using
    character context. Only chars with confidence below this threshold
    are corrected. Recommended: 0.7. Applies to all output formats.
    Text: appends [?] after each low-confidence word.
    JSON: adds \"low_confidence\": true to word objects.
    hOCR: adds CSS class \"low-confidence\" to ocrx_word spans.
    ALTO: adds QUALITY=\"needs review\" to String elements.

  --normalize-ja

    Convert full-width ASCII variants to half-width (Ａ→A, １→1, ！→!).
    Applied to all characters regardless of confidence. Can be combined
    with --post-correct-ja.

  -o, --output <path>

    Output file path (defaults to stdout)

  --confidence-heatmap <path>

    Save a confidence heatmap PNG to the given path. Each character's bounding
    box is overlaid with a color from red (low confidence) to green (high
    confidence) at 60% opacity. Can be combined with any output format.
    PDF input: not supported (image input only).

  -p, --png

    Output annotated copy of input image in PNG format

  --rec-model <path>

    Use a custom text recognition model

  --skip-text-pages

    Skip OCR for PDF pages that have a native text layer (Font resources).
    Useful for mixed PDFs containing both scanned and digital pages.
    Pages with text layers are noted to stderr even without this flag.

  --tables

    Detect table-like text layouts and output as CSV (default), markdown tables
    (with --markdown), or a \"tables\" array (with --json). PDF input: not supported.

  --tsv

    Output per-word data as TSV: text, left, top, right, bottom, confidence.
    PDF input prepends a page column. No header. Pipe-friendly.

  --region <x,y,w,h>

    Filter OCR output to text lines intersecting this pixel region.
    x,y = top-left corner; w,h = width and height.

  --version

    Display version info

Advanced options:

  (Note: These options are unstable and may change between releases)

  --beam

    Use beam search for decoding

  --debug

    Enable debug logging

  --text-line-images

    Export images of identified text lines

  --text-map

    Generate a text probability map for the input image

  --text-mask

    Generate a binary text mask for the input image

  --rec-per-box

    Recognize each detected box independently instead of merging boxes into
    lines first. Matches the PaddleOCR / RapidOCR pipeline; significantly
    better on tables and other multi-column layouts.

  --det-unclip <RATIO>

    Expand each detected box outwards by `area * RATIO / perimeter` before
    recognition, following PaddleOCR's DBPostProcess. PaddleOCR uses 1.5.
    Only meaningful for PaddleOCR DB detection models.
",
                    bin_name = parser.bin_name().unwrap_or("ocrs")
                );
                std::process::exit(0);
            }
            Long("version") => {
                println!("ocrs {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    // --model-dir sets default paths for rec-model and alphabet-file.
    if let Some(ref dir) = model_dir {
        if recognition_model.is_none() {
            recognition_model = Some(format!("{dir}/PP-OCRv5_server_rec_infer.onnx"));
        }
        if alphabet_file.is_none() {
            alphabet_file = Some(format!("{dir}/alphabet.txt"));
        }
        // Detection model is optional: only set if the file exists.
        if detection_model.is_none() {
            let det = format!("{dir}/PP-OCRv5_server_det_infer.onnx");
            if std::path::Path::new(&det).exists() {
                detection_model = Some(det);
            }
        }
    }

    // --lang sets CJK PP-OCRv5 model URLs; explicit flags override it.
    if lang {
        if detection_model.is_none() {
            detection_model = Some(CJK_DETECTION_MODEL.to_string());
        }
        if recognition_model.is_none() {
            recognition_model = Some(CJK_RECOGNITION_MODEL.to_string());
        }
        if alphabet.is_none() && alphabet_file.is_none() {
            alphabet = Some(CJK_ALPHABET.to_string());
        }
    }

    let image = values.pop_front();

    let stdin_is_pipe = !std::io::stdin().is_terminal();

    let input = match (clipboard, image, stdin_is_pipe) {
        (true, Some(_), _) => {
            return Err("cannot use both --clipboard and an image path".into());
        }
        (true, _, true) => {
            return Err("cannot use both --clipboard and stdin".into());
        }
        (true, None, false) => InputSource::Clipboard,
        (false, Some(path), _) => InputSource::File(path),
        (false, None, true) => InputSource::Stdin,
        (false, None, false) => {
            return Err("missing `<image>` arg (or use --clipboard / pipe to stdin)".into());
        }
    };

    Ok(Args {
        alphabet,
        alphabet_file,
        beam_search,
        debug,
        deskew,
        detection_model,
        input,
        low_confidence_mark,
        min_confidence,
        find_text,
        post_correct_ja,
        normalize_ja,
        region,
        model_dir,
        output_format,
        output_path,
        output_pdf,
        recognition_model,
        text_map,
        text_mask,
        text_line_images,
        rec_per_box,
        det_unclip,
        allowed_chars,
        tables,
        confidence_heatmap,
        auto_rotate,
        skip_text_pages,
    })
}

/// Default text detection model.
const DETECTION_MODEL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";

/// Default text recognition model.
const RECOGNITION_MODEL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

/// CJK (PP-OCRv5 server) detection model.
const CJK_DETECTION_MODEL: &str = "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_det_infer.onnx";

/// CJK (PP-OCRv5 server) recognition model.
const CJK_RECOGNITION_MODEL: &str = "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_rec_infer.onnx";

/// Embedded PP-OCRv5 alphabet (18,384 chars). Avoids a separate download for --lang users.
pub(crate) const CJK_ALPHABET: &str = include_str!("../../models/alphabet.txt");

/// Convert a decoded image into an HWC tensor.
/// Return true if axis-aligned rect `a` intersects the given [left, top, w, h] region.
fn rect_intersects(a: rten_imageproc::Rect, lx: i32, ty: i32, w: i32, h: i32) -> bool {
    (a.left() as i32) < lx + w
        && (a.right() as i32) > lx
        && (a.top() as i32) < ty + h
        && (a.bottom() as i32) > ty
}

fn image_to_tensor(image: image::DynamicImage) -> NdTensor<u8, 3> {
    let image = image.into_rgb8();
    let (width, height) = image.dimensions();
    NdTensor::from_data([height as usize, width as usize, 3], image.into_vec())
}

/// Load an image from a file path.
fn load_image_from_file(path: &str) -> anyhow::Result<NdTensor<u8, 3>> {
    image::open(path)
        .map(image_to_tensor)
        .with_context(|| format!("Failed to read image from {}", path))
}

/// Load an image from stdin.
fn load_image_from_stdin() -> anyhow::Result<NdTensor<u8, 3>> {
    let mut buf = Vec::new();
    std::io::stdin()
        .read_to_end(&mut buf)
        .context("Failed to read image from stdin")?;
    let image = image::load_from_memory(&buf).context("Failed to decode image from stdin")?;
    Ok(image_to_tensor(image))
}

/// Load an image from the system clipboard.
#[cfg(feature = "clipboard")]
fn load_image_from_clipboard() -> anyhow::Result<NdTensor<u8, 3>> {
    use arboard::Clipboard;

    let mut clipboard = Clipboard::new().context("Failed to access clipboard")?;

    let image_data = clipboard
        .get_image()
        .context("Failed to get image from clipboard. Is there an image copied?")?;

    // arboard returns RGBA, convert to RGB
    let rgba_bytes = image_data.bytes.into_owned();
    let rgb_bytes: Vec<u8> = rgba_bytes
        .chunks_exact(4)
        .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();

    Ok(NdTensor::from_data(
        [image_data.height, image_data.width, 3],
        rgb_bytes,
    ))
}

#[cfg(not(feature = "clipboard"))]
fn load_image_from_clipboard() -> anyhow::Result<NdTensor<u8, 3>> {
    Err(anyhow!(
        "ocrs was compiled without clipboard support. Use `cargo install ocrs-cli --features clipboard` to enable it."
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args().nth(1).as_deref() == Some("doctor") {
        doctor::run_doctor();
        return Ok(());
    }

    let args = parse_args()?;

    // Fetch and load ML models.
    let detection_model_src = args
        .detection_model
        .as_ref()
        .map_or(ModelSource::Url(DETECTION_MODEL), |path| {
            ModelSource::Path(path)
        });
    let detection_model = load_model(detection_model_src).with_context(|| {
        format!(
            "Failed to load text detection model from {}",
            detection_model_src
        )
    })?;

    let recognition_model_src = args
        .recognition_model
        .as_ref()
        .map_or(ModelSource::Url(RECOGNITION_MODEL), |path| {
            ModelSource::Path(path)
        });
    let recognition_model = load_model(recognition_model_src).with_context(|| {
        format!(
            "Failed to load text recognition model from {}",
            recognition_model_src
        )
    })?;

    // Resolve alphabet: --alphabet-file takes characters from a file, avoiding
    // shell escaping issues with large CJK character sets.
    let alphabet_from_file = args
        .alphabet_file
        .map(|path| {
            std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read alphabet file {}", path))
        })
        .transpose()?;
    let alphabet = alphabet_from_file.or(args.alphabet);

    // Initialize OCR engine.
    #[allow(clippy::needless_update)]
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        debug: args.debug,
        alphabet,
        decode_method: if args.beam_search {
            DecodeMethod::BeamSearch { width: 100 }
        } else {
            DecodeMethod::Greedy
        },
        allowed_chars: args.allowed_chars,
        rec_per_detected_rect: args.rec_per_box,
        detection_params: TextDetectorParams {
            unclip_ratio: args.det_unclip.unwrap_or(0.),
            ..Default::default()
        },
        ..Default::default()
    })?;

    // PDF input: extract page images, OCR each, write text + optional searchable PDF.
    #[cfg(not(target_arch = "wasm32"))]
    if let InputSource::File(ref file_path) = args.input {
        if file_path.to_lowercase().ends_with(".pdf") {
            let pdf_path = file_path.clone();
            let page_images = pdf::extract_page_images(&pdf_path)?;
            let mut all_lines: Vec<Option<ocrs_cjk::TextLine>> = Vec::new();
            let mut page_results: Vec<pdf::PageOcrResult> = Vec::new();

            for (page_idx, page) in page_images.into_iter().enumerate() {
                if page.has_text_layer {
                    if args.skip_text_pages {
                        eprintln!(
                            "note: page {} has text layer — skipping OCR (--skip-text-pages)",
                            page_idx + 1
                        );
                        continue;
                    } else {
                        eprintln!(
                            "note: page {} has text layer — OCRing image anyway \
                             (use --skip-text-pages to skip)",
                            page_idx + 1
                        );
                    }
                }
                // Destructure to allow tensor transforms while retaining other fields.
                let pdf::PageImage {
                    tensor: page_tensor,
                    width_pts,
                    height_pts,
                    has_text_layer: _,
                } = page;

                let page_tensor = if args.auto_rotate {
                    let (corrected, degrees) = deskew::auto_rotate(page_tensor.view());
                    if args.debug && degrees > 0 {
                        eprintln!(
                            "page {}: auto-rotate: applied {}° correction",
                            page_idx + 1,
                            degrees
                        );
                    }
                    corrected
                } else {
                    page_tensor
                };

                let page_tensor = if args.deskew {
                    let (corrected, angle) = deskew::deskew(page_tensor.view());
                    if args.debug {
                        eprintln!(
                            "page {}: deskew: detected {:.1}°, corrected",
                            page_idx + 1,
                            angle
                        );
                    }
                    corrected
                } else {
                    page_tensor
                };

                let [img_h, img_w, _] = page_tensor.shape();
                let img_src = ImageSource::from_tensor(page_tensor.view(), DimOrder::Hwc)?;
                let ocr_input = engine.prepare_input(img_src)?;
                let word_rects = engine.detect_words(&ocr_input)?;
                let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
                let text_lines = engine.recognize_text(&ocr_input, &line_rects)?;
                let text_lines = if let Some(t) = args.post_correct_ja {
                    post_correct::apply_ja(&text_lines, t)
                } else {
                    text_lines
                };
                let text_lines = if args.normalize_ja {
                    post_correct::normalize_ja(&text_lines)
                } else {
                    text_lines
                };
                let text_lines = if let Some([lx, ty, w, h]) = args.region {
                    text_lines
                        .into_iter()
                        .map(|line| {
                            line.filter(|l| rect_intersects(l.bounding_rect(), lx, ty, w, h))
                        })
                        .collect()
                } else {
                    text_lines
                };
                let text_lines = if let Some(ref q) = args.find_text {
                    text_lines
                        .into_iter()
                        .map(|line| line.filter(|l| l.to_string().contains(q.as_str())))
                        .collect()
                } else {
                    text_lines
                };
                page_results.push(pdf::PageOcrResult {
                    text_lines: text_lines.clone(),
                    image_hw: [img_h, img_w],
                    page_wh_pts: [width_pts, height_pts],
                });
                all_lines.extend(text_lines);
            }

            let write_str = |content: String| -> Result<(), Box<dyn Error>> {
                if let Some(ref p) = args.output_path {
                    std::fs::write(p, content.into_bytes())
                        .with_context(|| format!("Failed to write output to {p}"))?;
                } else {
                    println!("{}", content);
                }
                Ok(())
            };

            let page_infos: Vec<PageInfo> = page_results
                .iter()
                .map(|r| PageInfo {
                    image_hw: r.image_hw,
                    text_lines: &r.text_lines,
                    low_confidence_threshold: args.low_confidence_mark,
                })
                .collect();

            match args.output_format {
                OutputFormat::Text => {
                    write_str(format_text_output(&all_lines, args.low_confidence_mark))?
                }
                OutputFormat::Json => write_str(format_json_pdf_output(&pdf_path, &page_infos))?,
                OutputFormat::Hocr => write_str(format_hocr_pdf_output(&pdf_path, &page_infos))?,
                OutputFormat::Alto => write_str(format_alto_pdf_output(&page_infos))?,
                OutputFormat::Markdown => write_str(format_markdown_pdf_output(&page_infos))?,
                OutputFormat::Tsv => write_str(format_tsv_pdf_output(&page_infos))?,
                OutputFormat::Png => {
                    return Err("--png is not supported for PDF input".into());
                }
                OutputFormat::BlocksJson => {
                    return Err("--blocks-json is not supported for PDF input".into());
                }
                OutputFormat::FuriganaJson => {
                    return Err("--furigana-separate is not supported for PDF input".into());
                }
                OutputFormat::ReviewJson => {
                    return Err("--review-json is not supported for PDF input".into());
                }
            }

            if let Some(ref pdf_out) = args.output_pdf {
                pdf::build_searchable_pdf(&pdf_path, &page_results, pdf_out, args.min_confidence)?;
                if args.debug {
                    println!("Searchable PDF written to {pdf_out}");
                }
            }

            return Ok(());
        }
    }

    // Read image into HWC tensor.
    let (mut color_img, input_path): (NdTensor<u8, 3>, String) = match &args.input {
        InputSource::Clipboard => (load_image_from_clipboard()?, "<clipboard>".to_string()),
        InputSource::File(path) => (load_image_from_file(path)?, path.clone()),
        InputSource::Stdin => (load_image_from_stdin()?, "<stdin>".to_string()),
    };

    if args.auto_rotate {
        let (corrected, degrees) = deskew::auto_rotate(color_img.view());
        if args.debug && degrees > 0 {
            eprintln!("auto-rotate: applied {}° correction", degrees);
        }
        color_img = corrected;
    }

    if args.deskew {
        let (corrected, angle) = deskew::deskew(color_img.view());
        if args.debug {
            eprintln!("deskew: detected {:.1}°, corrected", angle);
        }
        color_img = corrected;
    }

    // Preprocess image for use with OCR engine.
    let color_img_source = ImageSource::from_tensor(color_img.view(), DimOrder::Hwc)?;
    let ocr_input = engine.prepare_input(color_img_source)?;

    if args.text_map || args.text_mask {
        let text_map = engine.detect_text_pixels(&ocr_input)?;
        let [height, width] = text_map.shape();
        let text_map = text_map.into_shape([1, height, width]);
        if args.text_map {
            write_image("text-map.png", text_map.view())?;
        }

        if args.text_mask {
            let threshold = engine.detection_threshold();
            let text_mask = text_map.map(|x| if *x > threshold { 1. } else { 0. });
            write_image("text-mask.png", text_mask.view())?;
        }
    }

    let word_rects = engine.detect_words(&ocr_input)?;

    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    if args.text_line_images {
        write_preprocessed_text_line_images(&ocr_input, &engine, &line_rects, "lines")?;
        // write_text_line_images(color_img.view(), &line_rects, "lines")?;
    }

    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;
    let line_texts = if let Some(t) = args.post_correct_ja {
        post_correct::apply_ja(&line_texts, t)
    } else {
        line_texts
    };
    let line_texts = if args.normalize_ja {
        post_correct::normalize_ja(&line_texts)
    } else {
        line_texts
    };
    let line_texts = if let Some([lx, ty, w, h]) = args.region {
        line_texts
            .into_iter()
            .map(|line| line.filter(|l| rect_intersects(l.bounding_rect(), lx, ty, w, h)))
            .collect()
    } else {
        line_texts
    };
    let line_texts = if let Some(ref q) = args.find_text {
        line_texts
            .into_iter()
            .map(|line| line.filter(|l| l.to_string().contains(q.as_str())))
            .collect()
    } else {
        line_texts
    };

    let detected_tables = if args.tables {
        table::detect_tables(&line_texts)
    } else {
        vec![]
    };

    let write_output_str = |content: String| -> Result<(), Box<dyn Error>> {
        if let Some(output_path) = &args.output_path {
            std::fs::write(output_path, content.into_bytes())
                .with_context(|| format!("Failed to write output to {}", output_path))?;
        } else {
            println!("{}", content);
        }
        Ok(())
    };

    match args.output_format {
        OutputFormat::Text => {
            let mut content = format_text_output(&line_texts, args.low_confidence_mark);
            if !detected_tables.is_empty() {
                content.push_str("\n\n");
                content.push_str(&table::format_tables_csv(&detected_tables));
            }
            write_output_str(content)?;
        }
        OutputFormat::Json => {
            let content = format_json_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: if args.tables {
                    Some(&detected_tables)
                } else {
                    None
                },
            });
            write_output_str(content)?;
        }
        OutputFormat::Png => {
            let png_args = GeneratePngArgs {
                img: color_img.view(),
                line_rects: &line_rects,
                text_lines: &line_texts,
            };
            let annotated_img = generate_annotated_png(png_args);
            let Some(output_path) = args.output_path else {
                return Err("Output path must be specified when generating annotated PNG".into());
            };
            write_image(&output_path, annotated_img.view())
                .with_context(|| format!("Failed to write output to {}", &output_path))?;
        }
        OutputFormat::Hocr => {
            let content = format_hocr_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: None,
            });
            write_output_str(content)?;
        }
        OutputFormat::Alto => {
            let content = format_alto_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: None,
            });
            write_output_str(content)?;
        }
        OutputFormat::Markdown => {
            let mut content = format_markdown_output(&line_texts, args.low_confidence_mark);
            if !detected_tables.is_empty() {
                content.push('\n');
                content.push_str(&table::format_tables_markdown(&detected_tables));
            }
            write_output_str(content)?;
        }
        OutputFormat::Tsv => {
            let content = format_tsv_output(&line_texts, args.low_confidence_mark);
            write_output_str(content)?;
        }
        OutputFormat::BlocksJson => {
            let content = format_blocks_json_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: None,
            });
            write_output_str(content)?;
        }
        OutputFormat::FuriganaJson => {
            let content = format_furigana_json_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: None,
            });
            write_output_str(content)?;
        }
        OutputFormat::ReviewJson => {
            let content = format_review_json_output(FormatJsonArgs {
                input_path: &input_path,
                input_hw: color_img.shape()[1..].try_into()?,
                text_lines: &line_texts,
                low_confidence_threshold: args.low_confidence_mark,
                tables: None,
            });
            write_output_str(content)?;
        }
    }

    if let Some(ref heatmap_path) = args.confidence_heatmap {
        let heatmap = generate_confidence_heatmap(GeneratePngArgs {
            img: color_img.view(),
            line_rects: &line_rects,
            text_lines: &line_texts,
        });
        write_image(heatmap_path, heatmap.view())
            .with_context(|| format!("Failed to write heatmap to {}", heatmap_path))?;
    }

    if args.debug {
        println!(
            "Found {} words, {} lines in image of size {}x{}",
            word_rects.len(),
            line_rects.len(),
            color_img.size(2),
            color_img.size(1),
        );
    }

    Ok(())
}
