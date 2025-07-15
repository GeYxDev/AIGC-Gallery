package dao.service;

import dao.entity.Follows;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

public interface FollowsService extends IService<Follows> {
    @Transactional
    Feedback getFollowerList(String account);
    @Transactional
    Feedback getFollowedList(String account);
    @Transactional
    Feedback isFollowed(String account, String followedAccount);
    @Transactional
    Feedback addFollowed(String account, String followedAccount);
    @Transactional
    Feedback cancelFollowed(String account, String followedAccount);
    @Transactional
    Feedback countFollower(String account);
    @Transactional
    Feedback countFollowed(String account);
}
