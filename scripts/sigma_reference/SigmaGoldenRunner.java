import ij.IJ;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ShortProcessor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Minimal headless adapter around the published Sigma_Filter_Plus bytecode.
 *
 * <p>This class deliberately lives in the unnamed package. The published plugin
 * also lives there, which lets this adapter read its package/protected kernel
 * fields without reflection. It bypasses only ImageJ's dialog, ROI, and stack
 * orchestration: filtering is performed by Sigma_Filter_Plus.doFiltering and
 * unsigned restoration is performed by ImageJ's ByteProcessor/ShortProcessor.
 */
public final class SigmaGoldenRunner {
    private SigmaGoldenRunner() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 9) {
            throw new IllegalArgumentException(
                    "Usage: SigmaGoldenRunner dtype width height radius sigma "
                            + "fraction outlier input.raw output.raw");
        }
        String dtype = args[0];
        int width = Integer.parseInt(args[1]);
        int height = Integer.parseInt(args[2]);
        double radius = Double.parseDouble(args[3]);
        double sigmaWidth = Double.parseDouble(args[4]);
        double minimumFraction = Double.parseDouble(args[5]);
        boolean outlierAware = Boolean.parseBoolean(args[6]);
        Path inputPath = Path.of(args[7]);
        Path outputPath = Path.of(args[8]);

        int pixelCount = Math.multiplyExact(width, height);
        byte[] rawInput = Files.readAllBytes(inputPath);
        float[] floatPixels = decode(dtype, rawInput, pixelCount);
        FloatProcessor filtered = new FloatProcessor(width, height, floatPixels);

        Sigma_Filter_Plus plugin = new Sigma_Filter_Plus();
        plugin.makeKernel(radius);
        int minimumPixelCount =
                (int) (plugin.kNPoints * minimumFraction + 0.999999);
        plugin.doFiltering(
                filtered,
                plugin.kRadius,
                plugin.lineRadius.clone(),
                sigmaWidth,
                minimumPixelCount,
                outlierAware);

        Files.write(outputPath, restore(dtype, width, height, filtered));
        System.out.printf(
                Locale.ROOT,
                "{\"imagej_version\":\"%s\",\"java_version\":\"%s\","
                        + "\"java_vendor\":\"%s\",\"os_name\":\"%s\","
                        + "\"os_arch\":\"%s\"}%n",
                jsonEscape(IJ.getVersion()),
                jsonEscape(System.getProperty("java.version")),
                jsonEscape(System.getProperty("java.vendor")),
                jsonEscape(System.getProperty("os.name")),
                jsonEscape(System.getProperty("os.arch")));
    }

    private static float[] decode(String dtype, byte[] raw, int pixelCount) {
        float[] result = new float[pixelCount];
        if (dtype.equals("uint8")) {
            if (raw.length != pixelCount) {
                throw new IllegalArgumentException("Unexpected uint8 input length");
            }
            for (int index = 0; index < pixelCount; index++) {
                result[index] = raw[index] & 0xff;
            }
            return result;
        }
        if (dtype.equals("uint16")) {
            if (raw.length != 2 * pixelCount) {
                throw new IllegalArgumentException("Unexpected uint16 input length");
            }
            ByteBuffer buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            for (int index = 0; index < pixelCount; index++) {
                result[index] = buffer.getShort() & 0xffff;
            }
            return result;
        }
        throw new IllegalArgumentException("Unsupported dtype: " + dtype);
    }

    private static byte[] restore(
            String dtype, int width, int height, FloatProcessor filtered) {
        if (dtype.equals("uint8")) {
            ByteProcessor restored = new ByteProcessor(width, height);
            restored.setPixels(0, filtered);
            return (byte[]) restored.getPixels();
        }
        if (dtype.equals("uint16")) {
            ShortProcessor restored = new ShortProcessor(width, height);
            restored.setPixels(0, filtered);
            short[] pixels = (short[]) restored.getPixels();
            ByteBuffer buffer =
                    ByteBuffer.allocate(2 * pixels.length).order(ByteOrder.LITTLE_ENDIAN);
            for (short pixel : pixels) {
                buffer.putShort(pixel);
            }
            return buffer.array();
        }
        throw new IllegalArgumentException("Unsupported dtype: " + dtype);
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
