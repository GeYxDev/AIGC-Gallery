package dao.entity;

import lombok.Data;

// Card展示数据
@Data
public class Cards {
    private Integer groupId;
    private String account;
    private String nickname;
    private String avatar;
    private String theme;
    private Integer likes;
    private Float aspectRatio;
    private String type;
    private String media;
}
