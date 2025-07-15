package func.controller;

import com.alibaba.fastjson2.JSON;
import dao.service.UsersService;
import func.service.CreateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import util.Feedback;

@RestController
@RequestMapping("/api")
public class CreateController {

    private final String baseRequestUrl = "http://127.0.0.1:5000/";

    private final CreateService createService;
    private final UsersService usersService;

    @Autowired
    public CreateController(CreateService createService, UsersService usersService) {
        this.createService = createService;
        this.usersService = usersService;
    }

    @PostMapping("/t2t")
    public String t2t(@RequestParam("account") String account, @RequestParam("password") String password,
                      @RequestParam("text") String text, @RequestParam("wordLimit") String wordLimit,
                      @RequestParam("textType") String textType, @RequestParam("prompt") String prompt) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        String requestUrl = baseRequestUrl + "creator/submitTextTask";
        return createService.submitTextTask(requestUrl, text, wordLimit, textType, prompt);
    }

    @PostMapping("/t2i")
    public String t2i(@RequestParam("account") String account, @RequestParam("password") String password,
                      @RequestParam("model") String model, @RequestParam("iteration") String iteration,
                      @RequestParam("style") String style, @RequestParam("word") String word) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        String requestUrl = baseRequestUrl + "creator/submitMediaTask";
        return createService.submitMediaTask(requestUrl, "image", model, iteration, style, word);
    }

    @PostMapping("/t2v")
    public String t2v(@RequestParam("account") String account, @RequestParam("password") String password,
                      @RequestParam("model") String model, @RequestParam("iteration") String iteration,
                      @RequestParam("duration") String duration, @RequestParam("word") String word) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        String requestUrl = baseRequestUrl + "creator/submitMediaTask";
        return createService.submitMediaTask(requestUrl, "video", model, iteration, duration, word);
    }

    @PostMapping("/enquire")
    public String t2i(@RequestParam("uuid") String uuid) {
        String requestUrl = baseRequestUrl + "creator/enquireMediaTask";
        return createService.enquireMediaTask(requestUrl, uuid);
    }
}
