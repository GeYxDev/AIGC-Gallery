package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.entity.Users;
import dao.service.UsersService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/identity")
public class UsersController {

    private final UsersService usersService;

    @Autowired
    public UsersController(UsersService usersService) {
        this.usersService = usersService;
    }

    @PostMapping("/login")
    public String handleLoginRequest(@RequestParam("account") String account, @RequestParam("password") String password) {
        return JSON.toJSONString(usersService.userLogin(account, password));
    }

    @PostMapping("/register")
    public String handleRegisterRequest(@RequestParam("nickname") String nickname, @RequestParam("avatar") String avatar,
                                        @RequestParam("account") String account, @RequestParam("password") String password) {
        Users user = new Users();
        user.setNickname(nickname);
        user.setAvatar(avatar);
        user.setAccount(account);
        user.setPassword(password);
        return JSON.toJSONString(usersService.userCreate(user));
    }

    @PostMapping("/modifyPassword")
    public String modifyPassword(@RequestParam("account") String account,
                                 @RequestParam("oldPassword") String oldPassword, @RequestParam("newPassword") String newPassword) {
        return JSON.toJSONString(usersService.changePassword(account, oldPassword, newPassword));
    }

    @PostMapping("/modifyAvatar")
    public String modifyAvatar(@RequestParam("account") String account,
                                 @RequestParam("password") String password, @RequestParam("avatar") String avatar) {
        return JSON.toJSONString(usersService.changeUserAvatar(account, password, avatar));
    }

    @PostMapping("/modifyNickname")
    public String modifyNickname(@RequestParam("account") String account,
                               @RequestParam("password") String password, @RequestParam("nickname") String nickname) {
        return JSON.toJSONString(usersService.changeUserNickname(account, password, nickname));
    }
}
