package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.service.GroupsService;
import dao.service.UsersService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import util.Feedback;
import util.FeedbackException;

@RestController
@RequestMapping("/post")
public class GroupsController {

    private final GroupsService groupsService;
    private final UsersService usersService;

    @Autowired
    GroupsController(GroupsService groupsService, UsersService usersService) {
        this.groupsService = groupsService;
        this.usersService = usersService;
    }

    @PostMapping("/getMoments")
    public String getMoments(@RequestParam("account") String account,
                             @RequestParam("page") int page, @RequestParam("size") int size) {
        return JSON.toJSONString(groupsService.getMomentByAccount(account, page, size));
    }

    @PostMapping("/getIndividualMoments")
    public String getIndividualMoments(@RequestParam("account") String account,
                                       @RequestParam("page") int page, @RequestParam("size") int size) {
        return JSON.toJSONString(groupsService.getIndividualMoment(account, page, size));
    }

    @PostMapping("/addLike")
    public String addLike(@RequestParam("groupId") int groupId) {
        return JSON.toJSONString(groupsService.addGroupLikes(groupId));
    }

    @PostMapping("/deleteMoment")
    public String deleteMoment(@RequestParam("account") String account,
                               @RequestParam("password") String password, @RequestParam("groupId") int groupId) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        return JSON.toJSONString(groupsService.deleteGroupByGroupId(account, groupId));
    }

    @PostMapping("/releaseMoment")
    public String releaseMoment(@RequestParam("account") String account, @RequestParam("password") String password,
                                @RequestParam("theme") String theme, @RequestParam("content") String content,
                                @RequestParam("artList") String artList, @RequestParam("artType") String artType) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        try {
            return JSON.toJSONString(groupsService.submitMoment(account, theme, content, artList, artType));
        } catch (FeedbackException e) {
            return JSON.toJSONString(new Feedback(e.isSuccess(), e.getResult()));
        }
    }
}
