package dao.entity;

import lombok.Data;

// Details展示数据
@Data
public class Details {
    private Integer workId;
    private Integer mediaWidth;
    private Integer mediaHeight;
    private String modelType;
    private Integer iteration;
    private String videoLink;
    private String image;
}
