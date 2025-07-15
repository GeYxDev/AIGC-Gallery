package dao.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Getter;
import lombok.Setter;

import java.io.Serial;
import java.time.LocalDateTime;
import java.io.Serializable;

@Setter
@Getter
@TableName("Groups")
public class Groups implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "group_id", type = IdType.AUTO)
    private Integer groupId;

    private String account;

    private String theme;

    private String content;

    private Integer likes;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;


    @Override
    public String toString() {
        return "Groups{" +
        "groupId=" + groupId +
        ", account=" + account +
        ", theme=" + theme +
        ", content=" + content +
        ", likes=" + likes +
        ", createTime=" + createTime +
        "}";
    }
}
