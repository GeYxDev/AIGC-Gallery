package util;


import lombok.Getter;
import lombok.Setter;

// 返回错误原因
@Setter
@Getter
public class FeedbackException extends RuntimeException {
    private final boolean success;
    private final String result;

    public FeedbackException(boolean success, String result) {
        this.success = success;
        this.result = result;
    }
}
