package dao.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;

import java.io.Serial;
import java.io.Serializable;

@Setter
@Getter
@TableName("Follows")
public class Follows implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private String followerAccount;

    private String followedAccount;


    @Override
    public String toString() {
        return "Follows{" +
        "followerAccount=" + followerAccount +
        ", followedAccount=" + followedAccount +
        "}";
    }
}
