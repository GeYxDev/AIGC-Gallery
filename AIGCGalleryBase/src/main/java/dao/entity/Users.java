package dao.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Getter;
import lombok.Setter;

import java.io.Serial;
import java.io.Serializable;

@Setter
@Getter
@TableName("Users")
public class Users implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "account", type = IdType.INPUT)
    private String account;

    private String password;

    private String nickname;

    private String avatar;


    @Override
    public String toString() {
        return "Users{" +
        "account=" + account +
        ", password=" + password +
        ", nickname=" + nickname +
        ", avatar=" + avatar +
        "}";
    }
}
