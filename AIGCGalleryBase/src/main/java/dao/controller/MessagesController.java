package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.service.MessagesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/detail")
public class MessagesController {

    private final MessagesService messagesService;

    @Autowired
    public MessagesController(MessagesService messagesService) {
        this.messagesService = messagesService;
    }

    @PostMapping("/getComments")
    public String getComments(@RequestParam("groupId") int groupId) {
        return JSON.toJSONString(messagesService.getCommentList(groupId));
    }
}
