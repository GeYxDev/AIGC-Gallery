package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.entity.Comments;
import dao.service.CommentsService;
import dao.service.UsersService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import util.Feedback;

@RestController
@RequestMapping("/comment")
public class CommentsController {

    private final CommentsService commentsService;
    private final UsersService usersService;

    @Autowired
    public CommentsController(CommentsService commentsService, UsersService usersService) {
        this.commentsService = commentsService;
        this.usersService = usersService;
    }

    @PostMapping("/addComment")
    public String addFollow(@RequestParam("account") String account, @RequestParam("password") String password,
                            @RequestParam("groupId") int groupId, @RequestParam("content") String content) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        Comments comments = new Comments();
        comments.setGroupId(groupId);
        comments.setAccount(account);
        comments.setContent(content);
        return JSON.toJSONString(commentsService.addComment(comments));
    }
}
