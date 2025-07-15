package dao.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Getter;
import lombok.Setter;

import java.io.Serial;
import java.time.LocalDateTime;
import java.io.Serializable;

@Setter
@Getter
@TableName("Comments")
public class Comments implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "comment_id", type = IdType.AUTO)
    private Integer commentId;

    private Integer groupId;

    private String account;

    private String content;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;


    @Override
    public String toString() {
        return "Comments{" +
        "commentId=" + commentId +
        ", groupId=" + groupId +
        ", account=" + account +
        ", content=" + content +
        ", createTime=" + createTime +
        "}";
    }
}
