package func.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

@RestController
public class videoTransmitController {

    private final ResourceLoader resourceLoader;

    @Autowired
    public videoTransmitController(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    @GetMapping("/video/{videoFileName}")
    public ResponseEntity<StreamingResponseBody> getVideo
            (@PathVariable String videoFileName, @RequestHeader(value = "Range", required = false) String rangeHeader) {
        Resource resource = resourceLoader.getResource("file:src/main/video/");
        String videoDir;
        try {
            videoDir = resource.getFile().getAbsolutePath();
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(outputStream -> outputStream.write("获取视频目录失败".getBytes(StandardCharsets.UTF_8)));
        }
        String videoFilePath = videoDir + File.separator + videoFileName;
        Resource videoResource = new FileSystemResource(videoFilePath);
        if (!videoResource.exists()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(outputStream -> outputStream.write("视频文件不存在".getBytes(StandardCharsets.UTF_8)));
        }
        long contentLength;
        try {
            contentLength = Files.size(Path.of(videoFilePath));
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(outputStream -> outputStream.write("获取视频大小失败".getBytes(StandardCharsets.UTF_8)));
        }
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaTypeFactory.getMediaType(videoResource).orElse(MediaType.APPLICATION_OCTET_STREAM));
        if (rangeHeader == null) {
            headers.setContentLength(contentLength);
            return ResponseEntity.ok().headers(headers)
                    .body(outputStream -> Files.copy(Path.of(videoFilePath), outputStream));
        }
        String[] ranges = rangeHeader.split("=")[1].split("-");
        long rangeStart = Long.parseLong(ranges[0]);
        long rangeEnd = ranges.length > 1 ? Long.parseLong(ranges[1]) : contentLength - 1;
        if (rangeEnd >= contentLength) {
            rangeEnd = contentLength - 1;
        }
        long rangeLength = rangeEnd - rangeStart + 1;
        headers.set(HttpHeaders.CONTENT_RANGE, "bytes " + rangeStart + "-" + rangeEnd + "/" + contentLength);
        headers.setContentLength(rangeLength);
        headers.set(HttpHeaders.ACCEPT_RANGES, "bytes");
        return ResponseEntity.status(HttpStatus.PARTIAL_CONTENT).headers(headers).body(outputStream -> {
            try (var inputStream = Files.newInputStream(Path.of(videoFilePath))) {
                inputStream.skipNBytes(rangeStart);
                byte[] buffer = new byte[1024 * 1024];
                long remaining = rangeLength;
                while (remaining > 0) {
                    int read = inputStream.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                    if (read == -1) {
                        break;
                    }
                    outputStream.write(buffer, 0, read);
                    remaining -= read;
                }
            }
        });
    }
}
