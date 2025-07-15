package util;


import lombok.Getter;
import lombok.Setter;

// 返回操作结果
@Setter
@Getter
public class Feedback {
    private boolean success;
    private Object result;

    public Feedback(boolean success, Object result) {
        this.success = success;
        this.result = result;
    }
}
