package func.service;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import util.Feedback;

import java.util.UUID;

@Service
public class CreateService {

    private final RestTemplate restTemplate;

    @Autowired
    public CreateService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public String submitMediaTask(String requestUrl, String mediaType, String model,
                                  String iteration, String optionalParameter, String word) {
        String taskId = UUID.randomUUID().toString();
        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 构建请求体参数
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("taskId", taskId);
        body.add("prompt", word);
        body.add("modelType", model);
        body.add("createType", mediaType);
        body.add("iteration", iteration);
        if (mediaType.equals("image")) {
            body.add("style", optionalParameter);
        } else if (mediaType.equals("video")) {
            body.add("duration", optionalParameter);
        } else {
            return JSON.toJSONString(new Feedback(false, "生成类型错误"));
        }
        // 封装请求实体
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        try {
            // 发送并提交生成任务
            ResponseEntity<String> response = restTemplate.postForEntity(requestUrl, requestEntity, String.class);
            // 处理响应结果
            if (response.getStatusCode().is2xxSuccessful()) {
                // 请求成功时解析响应内容
                String responseBody = response.getBody();
                if (responseBody != null) {
                    JSONObject jsonResponse = JSON.parseObject(responseBody);
                    if (jsonResponse.getBooleanValue("success")) {
                        // 生成任务成功提交
                        return JSON.toJSONString(new Feedback(true, taskId));
                    }
                }
                return JSON.toJSONString(new Feedback(false, "作品生成失败"));
            } else {
                // 请求成功时返回服务未启动错误
                return JSON.toJSONString(new Feedback(false, "服务启动失败"));
            }
        } catch (Exception e) {
            return JSON.toJSONString(new Feedback(false, "作品生成失败" + e.getMessage()));
        }
    }

    public String enquireMediaTask(String requestUrl, String uuid) {
        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 构建请求体参数
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("taskId", uuid);
        // 封装请求实体
        HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, headers);
        try {
            // 发送并提交生成任务状态查询
            ResponseEntity<String> response = restTemplate.postForEntity(requestUrl, requestEntity, String.class);
            // 处理响应结果
            if (response.getStatusCode().is2xxSuccessful()) {
                // 请求成功时解析响应内容
                String responseBody = response.getBody();
                if (responseBody != null) {
                    JSONObject jsonResponse = JSON.parseObject(responseBody);
                    if (jsonResponse.get("result") instanceof JSONObject) {
                        Feedback feedback = new Feedback(
                                jsonResponse.getBooleanValue("success"),
                                jsonResponse.getJSONObject("result")
                        );
                        return JSON.toJSONString(feedback);
                    } else {
                        Feedback feedback = new Feedback(
                                jsonResponse.getBooleanValue("success"),
                                jsonResponse.getString("result")
                        );
                        return JSON.toJSONString(feedback);
                    }
                } else {
                    return JSON.toJSONString(new Feedback(false, "作品生成失败"));
                }
            } else {
                // 请求成功时返回服务未启动错误
                return JSON.toJSONString(new Feedback(false, "服务响应异常"));
            }
        } catch (Exception e) {
            return JSON.toJSONString(new Feedback(false, "作品生成失败" + e.getMessage()));
        }
    }

    public String submitTextTask(String requestUrl, String text, String wordLimit, String textType, String prompt) {
        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 构建请求体参数
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("text", text);
        body.add("wordLimit", wordLimit);
        body.add("textType", textType);
        body.add("prompt", prompt);
        // 封装请求实体
        HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(body, headers);
        try {
            // 发送并提交生成任务
            ResponseEntity<String> response = restTemplate.postForEntity(requestUrl, requestEntity, String.class);
            // 处理响应结果
            if (response.getStatusCode().is2xxSuccessful()) {
                // 请求成功时解析响应内容
                String responseBody = response.getBody();
                if (responseBody != null) {
                    JSONObject jsonResponse = JSON.parseObject(responseBody);
                    if (jsonResponse.getBooleanValue("success")) {
                        // 生成任务结果成功返回
                        return JSON.toJSONString(new Feedback(true, jsonResponse.getString("result")));
                    }
                }
                return JSON.toJSONString(new Feedback(false, "作品生成失败"));
            } else {
                // 请求成功时返回服务未启动错误
                return JSON.toJSONString(new Feedback(false, "服务启动失败"));
            }
        } catch (Exception e) {
            return JSON.toJSONString(new Feedback(false, "作品生成失败" + e.getMessage()));
        }
    }
}
