package dao.service;

import dao.entity.Users;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface UsersService extends IService<Users> {
    @Transactional
    Feedback userVerify(String account, String password);
    @Transactional
    Feedback userCreate(Users users);
    @Transactional
    Feedback userLogin(String account, String password);
    @Transactional
    Feedback changePassword(String account, String password, String newPassword);
    @Transactional
    Feedback changeUserAvatar(String account, String password, String avatar);
    @Transactional
    Feedback changeUserNickname(String account, String password, String nickname);
    @Transactional
    Feedback getUserInfo(String account);
    @Transactional
    Feedback changeUserInfo(Users users);
}
