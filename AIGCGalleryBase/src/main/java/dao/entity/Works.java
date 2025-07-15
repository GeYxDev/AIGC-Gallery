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
@TableName("Works")
public class Works implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "work_id", type = IdType.AUTO)
    private Integer workId;

    private Integer groupId;

    private Float aspectRatio;

    private String type;

    private String modelType;

    private Integer mediaWidth;

    private Integer mediaHeight;

    private Integer iteration;


    @Override
    public String toString() {
        return "Works{" +
        "workId=" + workId +
        ", groupId=" + groupId +
        ", aspectRatio=" + aspectRatio +
        ", type=" + type +
        ", modelType=" + modelType +
        ", mediaWidth=" + mediaWidth +
        ", mediaHeight=" + mediaHeight +
        ", iteration=" + iteration +
        "}";
    }
}
