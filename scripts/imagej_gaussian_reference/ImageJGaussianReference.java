import ij.IJ;
import ij.plugin.filter.GaussianBlur;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import java.util.Arrays;

/** Independent synthetic references produced by unmodified ImageJ 1.54p. */
public class ImageJGaussianReference {
    static String values(ImageProcessor ip) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < ip.getPixelCount(); i++) {
            if (i > 0) out.append(',');
            if (ip instanceof FloatProcessor) out.append(Float.toString(ip.getf(i)));
            else out.append(ip.get(i));
        }
        return out.append(']').toString();
    }
    public static void main(String[] args) {
        StringBuilder json = new StringBuilder("{\"imagej_version\":\"" + IJ.getVersion() + "\",\"cases\":[");
        boolean first = true;
        int[][] shapes = {{1, 1}, {2, 3}, {7, 9}};
        double[] sigmas = {0.0, 0.5, 1.5, 8.5};
        for (String dtype : new String[]{"uint8", "uint16", "float32"}) {
            for (int[] shape : shapes) {
                for (double sigma : sigmas) {
                    int h = shape[0], w = shape[1], n = h * w;
                    ImageProcessor ip;
                    if (dtype.equals("uint8")) {
                        byte[] pixels = new byte[n];
                        for (int i = 0; i < n; i++) pixels[i] = (byte)((i * 31 + 7) % 256);
                        pixels[0] = (byte)255;
                        ip = new ByteProcessor(w, h, pixels);
                    } else if (dtype.equals("uint16")) {
                        short[] pixels = new short[n];
                        for (int i = 0; i < n; i++) pixels[i] = (short)((i * 8111 + 997) % 65536);
                        pixels[0] = (short)65535;
                        ip = new ShortProcessor(w, h, pixels, null);
                    } else {
                        float[] pixels = new float[n];
                        for (int i = 0; i < n; i++) pixels[i] = (float)(i * 0.137 - 13.0);
                        pixels[0] = 1000.125f;
                        ip = new FloatProcessor(w, h, pixels);
                    }
                    String input = values(ip);
                    if (sigma > 0) new GaussianBlur().blurGaussian(ip, sigma, sigma, dtype.equals("uint8") ? 0.002 : 0.0002);
                    if (!first) json.append(',');
                    first = false;
                    json.append("{\"id\":\"").append(dtype).append('_').append(h).append('x').append(w).append('_').append(sigma)
                        .append("\",\"dtype\":\"").append(dtype).append("\",\"shape\":[").append(h).append(',').append(w)
                        .append("],\"sigma\":").append(sigma).append(",\"input_values\":").append(input)
                        .append(",\"output_values\":").append(values(ip)).append('}');
                }
            }
        }
        // Large enough to exercise ImageJ's distinct interior arithmetic.
        float[] negativeZero = new float[41 * 41];
        Arrays.fill(negativeZero, -0.0f);
        ImageProcessor ip = new FloatProcessor(41, 41, negativeZero);
        String input = values(ip);
        new GaussianBlur().blurGaussian(ip, 1.5, 1.5, 0.0002);
        json.append(",{\"id\":\"float32_41x41_1.5_negative_zero\",\"dtype\":\"float32\",\"shape\":[41,41],\"sigma\":1.5,\"input_values\":")
            .append(input).append(",\"output_values\":").append(values(ip)).append('}');
        System.out.println(json.append("]}"));
    }
}
