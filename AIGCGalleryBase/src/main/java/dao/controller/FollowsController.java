package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.service.FollowsService;
import dao.service.UsersService;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import util.Feedback;

@RestController
@RequestMapping("/follow")
public class FollowsController {

    private final FollowsService followsService;
    private final UsersService usersService;

    @Autowired
    public FollowsController(FollowsService followsService, UsersService usersService) {
        this.followsService = followsService;
        this.usersService = usersService;
    }

    @Setter
    @Getter
    public static class FollowCountResponse {
        private long followerNum;
        private long followedNum;

        public FollowCountResponse(long followerNum, long followedNum) {
            this.followerNum = followerNum;
            this.followedNum = followedNum;
        }
    }

    @PostMapping("/getFollowCount")
    public String getFollowCount(@RequestParam("account") String account) {
        Feedback countFollowerResult = followsService.countFollower(account);
        Feedback countFollowedResult = followsService.countFollowed(account);
        long countFollowerNum = countFollowerResult.isSuccess() ? (long) countFollowerResult.getResult() : -1;
        long countFollowedNum = countFollowedResult.isSuccess() ? (long) countFollowedResult.getResult() : -1;
        FollowCountResponse followCountResponse = new FollowCountResponse(countFollowerNum, countFollowedNum);
        return JSON.toJSONString(followCountResponse);
    }

    @PostMapping("/isFollowed")
    public String isFollowed(@RequestParam("account") String account,
                             @RequestParam("followedAccount") String followedAccount) {
        Feedback isFollowedResult = followsService.isFollowed(account, followedAccount);
        return JSON.toJSONString(isFollowedResult);
    }

    @PostMapping("/addFollow")
    public String addFollow(@RequestParam("account") String account,
                            @RequestParam("password") String password,
                            @RequestParam("followedAccount") String followedAccount) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        return JSON.toJSONString(followsService.addFollowed(account, followedAccount));
    }

    @PostMapping("/cancelFollow")
    public String cancelFollow(@RequestParam("account") String account,
                            @RequestParam("password") String password,
                            @RequestParam("followedAccount") String followedAccount) {
        Feedback isPassVerify =  usersService.userVerify(account, password);
        if (!isPassVerify.isSuccess()) {
            isPassVerify.setResult("用户验证失败");
            return JSON.toJSONString(isPassVerify);
        }
        return JSON.toJSONString(followsService.cancelFollowed(account, followedAccount));
    }
}
