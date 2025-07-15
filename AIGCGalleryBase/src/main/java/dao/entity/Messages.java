package dao.entity;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class Messages {
    private Integer commentId;
    private String nickname;
    private String avatar;
    private String content;
    private LocalDateTime createTime;
}
