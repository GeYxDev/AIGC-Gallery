package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.github.houbb.sensitive.word.core.SensitiveWordHelper;
import dao.entity.Users;
import dao.mapper.UsersMapper;
import dao.service.UsersService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.Feedback;

@Service
public class UsersServiceImpl extends ServiceImpl<UsersMapper, Users> implements UsersService {

    // 所有方法的身份验证自行解决

    @Override
    public Feedback userVerify(String account, String password) {
        // 用户身份验证
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (storedUser.getPassword().equals(password)) {
                    return new Feedback(true, "用户验证成功");
                } else {
                    return new Feedback(false, "密码错误");
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "用户验证失败" + e.getMessage());
        }
    }

    @Override
    public Feedback userCreate(Users users) {
        try {
            if (getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, users.getAccount())) != null) {
                return new Feedback(false, "账号已存在");
            }
            users.setNickname(SensitiveWordHelper.replace(users.getNickname()));
            boolean saveResult = save(users);
            if (saveResult) {
                return new Feedback(true, "用户创建成功");
            } else {
                return new Feedback(false, "用户创建失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "用户创建失败" + e.getMessage());
        }
    }

    @Override
    public Feedback userLogin(String account, String password) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (!storedUser.getPassword().equals(password)) {
                    return new Feedback(false, "密码错误");
                } else {
                    storedUser.setPassword(null);
                    return new Feedback(true, storedUser);
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "登录失败" + e.getMessage());
        }
    }

    @Override
    public Feedback changePassword(String account, String password, String newPassword) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (!storedUser.getPassword().equals(password)) {
                    return new Feedback(false, "旧密码错误");
                } else {
                    storedUser.setPassword(newPassword);
                    boolean updateResult = updateById(storedUser);
                    if (updateResult) {
                        return new Feedback(true, "密码修改成功");
                    } else {
                        return new Feedback(false, "密码修改失败, 数据库更新异常");
                    }
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "密码修改失败" + e.getMessage());
        }
    }

    @Override
    public Feedback changeUserAvatar(String account, String password, String avatar) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (!storedUser.getPassword().equals(password)) {
                    return new Feedback(false, "密码错误");
                } else {
                    storedUser.setAvatar(avatar);
                    boolean updateResult = updateById(storedUser);
                    if (updateResult) {
                        return new Feedback(true, "头像修改成功");
                    } else {
                        return new Feedback(false, "头像修改失败, 数据库更新异常");
                    }
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "头像修改失败" + e.getMessage());
        }
    }

    @Override
    public Feedback changeUserNickname(String account, String password, String nickname) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (!storedUser.getPassword().equals(password)) {
                    return new Feedback(false, "密码错误");
                } else {
                    storedUser.setNickname(nickname);
                    boolean updateResult = updateById(storedUser);
                    if (updateResult) {
                        return new Feedback(true, "昵称修改成功");
                    } else {
                        return new Feedback(false, "昵称修改失败, 数据库更新异常");
                    }
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "昵称修改失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getUserInfo(String account) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, account));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                storedUser.setPassword(null);
                return new Feedback(true, storedUser);
            }
        } catch (Exception e) {
            return new Feedback(false, "用户信息拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback changeUserInfo(Users users) {
        try {
            Users storedUser = getOne(new LambdaQueryWrapper<Users>().eq(Users::getAccount, users.getAccount()));
            if (storedUser == null) {
                return new Feedback(false, "用户不存在");
            } else {
                if (storedUser.getPassword().equals(users.getPassword())) {
                    return new Feedback(false, "密码错误");
                } else {
                    storedUser.setNickname(users.getNickname());
                    storedUser.setAvatar(users.getAvatar());
                    boolean updateResult = updateById(storedUser);
                    if (updateResult) {
                        storedUser.setPassword(null);
                        return new Feedback(true, storedUser);
                    } else {
                        return new Feedback(false, "信息修改失败, 数据库更新异常");
                    }
                }
            }
        } catch (Exception e) {
            return new Feedback(false, "信息修改失败" + e.getMessage());
        }
    }
}
