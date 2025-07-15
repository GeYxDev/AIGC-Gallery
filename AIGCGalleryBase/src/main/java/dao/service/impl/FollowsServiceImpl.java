package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import dao.entity.Follows;
import dao.mapper.FollowsMapper;
import dao.service.FollowsService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.Feedback;

@Service
public class FollowsServiceImpl extends ServiceImpl<FollowsMapper, Follows> implements FollowsService {

    @Override
    public Feedback getFollowerList(String account) {
        try {
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.select(Follows::getFollowerAccount).eq(Follows::getFollowedAccount, account);
            return new Feedback(true, list(lambdaQueryWrapper));
        } catch (Exception e) {
            return new Feedback(false, "粉丝列表查询失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getFollowedList(String account) {
        try {
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.select(Follows::getFollowedAccount).eq(Follows::getFollowerAccount, account);
            return new Feedback(true, list(lambdaQueryWrapper));
        } catch (Exception e) {
            return new Feedback(false, "关注列表查询失败" + e.getMessage());
        }
    }

    @Override
    public Feedback isFollowed(String account, String followedAccount) {
        try {
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Follows::getFollowerAccount, account)
                    .eq(Follows::getFollowedAccount, followedAccount);
            return new Feedback(true, count(lambdaQueryWrapper) > 0);
        } catch (Exception e) {
            return new Feedback(false, "查询失败" + e.getMessage());
        }
    }

    @Override
    public Feedback addFollowed(String account, String followedAccount) {
        try {
            // 用户身份交由上层解决
            Follows follows = new Follows();
            follows.setFollowerAccount(account);
            follows.setFollowedAccount(followedAccount);
            boolean saveResult = save(follows);
            if (saveResult) {
                return new Feedback(true, "关注成功");
            } else {
                return new Feedback(false, "关注失败，数据库插入错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "关注失败" + e.getMessage());
        }
    }

    @Override
    public Feedback cancelFollowed(String account, String followedAccount) {
        try {
            // 用户身份交由上层解决
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Follows::getFollowerAccount, account)
                    .eq(Follows::getFollowedAccount, followedAccount);
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "取消关注成功");
            } else {
                return new Feedback(false, "取消关注失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "取消关注失败" + e.getMessage());
        }
    }

    @Override
    public Feedback countFollower(String account) {
        try {
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Follows::getFollowedAccount, account);
            long followerCount = count(lambdaQueryWrapper);
            return new Feedback(true, followerCount);
        } catch (Exception e) {
            return new Feedback(false, "粉丝数统计失败" + e.getMessage());
        }
    }

    @Override
    public Feedback countFollowed(String account) {
        try {
            LambdaQueryWrapper<Follows> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Follows::getFollowerAccount, account);
            long followedCount = count(lambdaQueryWrapper);
            return new Feedback(true, followedCount);
        } catch (Exception e) {
            return new Feedback(false, "关注数统计失败" + e.getMessage());
        }
    }
}
