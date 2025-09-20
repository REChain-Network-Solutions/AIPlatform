<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Errors\ErrorCode;

abstract class AbstractTestApiRequestException extends ApiExceptionTest
{
    /**
     * @dataProvider descriptionDataProvider
     */
    public function testDescription(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        $tmp = json_decode($body, true);
        if (empty($tmp['description'])) {
            self::assertNull($instance->getError()->getDescription());
        } else {
            self::assertEquals($tmp['description'], $instance->getError()->getDescription());
        }
    }

    public static function descriptionDataProvider(): array
    {
        return [
            ['{}'],
            ['{"description":"test"}'],
            ['{"description":"У попа была собака"}'],
        ];
    }

    /**
     * @dataProvider codeDataProvider
     */
    public function testCode(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        $tmp = json_decode($body, true);
        if (empty($tmp['code']) || !in_array($tmp['code'], ErrorCode::getValidValues(), true)) {
            self::assertEquals('unknown', $instance->getError()->getCode());
        } else {
            self::assertEquals($tmp['code'], $instance->getError()->getCode());
        }
    }

    public static function codeDataProvider(): array
    {
        return [
            ['{}'],
            ['{"code":"123"}'],
            ['{"code":"server_error"}'],
            ['{"code":"internal_server_error"}'],
            ['{"code":"error"}'],
            ['{"code":"invalid_credentials"}'],
        ];
    }

    /**
     * @dataProvider parameterDataProvider
     */
    public function testParameter(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        $tmp = json_decode($body, true);
        if (empty($tmp['parameter'])) {
            self::assertNull($instance->getError()->getParameter());
        } else {
            self::assertEquals($tmp['parameter'], $instance->getError()->getParameter());
        }
    }

    public static function parameterDataProvider(): array
    {
        return [
            ['{}'],
            ['{"parameter":"parameter_name"}'],
            ['{"parameter":null}'],
        ];
    }

    /**
     * @dataProvider retryAfterDataProvider
     */
    public function testRetryAfter(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        $tmp = json_decode($body, true);
        if (empty($tmp['retry_after'])) {
            self::assertNull($instance->retryAfter);
        } else {
            self::assertEquals($tmp['retry_after'], $instance->retryAfter);
            self::assertEquals($tmp['retry_after'], $instance->getError()->getRetryAfter());
        }
    }

    public static function retryAfterDataProvider(): array
    {
        return [
            ['{}'],
            ['{"retry_after":-20}'],
            ['{"retry_after":123}'],
        ];
    }

    /**
     * @dataProvider typeDataProvider
     */
    public function testType(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        $tmp = json_decode($body, true);
        if (empty($tmp['type'])) {
            self::assertNull($instance->type);
            self::assertNull($instance->getError()->getType());
        } else {
            self::assertEquals($tmp['type'], $instance->type);
            self::assertEquals($tmp['type'], $instance->getError()->getType());
        }
    }

    public static function typeDataProvider(): array
    {
        return [
            ['{}'],
            ['{"type":"server_error"}'],
            ['{"type":"error"}'],
        ];
    }

    /**
     * @dataProvider messageDataProvider
     *
     * @param mixed $body
     */
    public function testMessage($body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);

        $tmp = json_decode($body, true);
        $message = '';
        if (!empty($tmp['description'])) {
            $message = $tmp['description'] . '. ';
        }
        if (empty($tmp['code']) || !in_array($tmp['code'], ErrorCode::getValidValues(), true)) {
            $message .= 'Error code: unknown. ';
        } else {
            $message .= 'Error code: ' . $tmp['code'] . '. ';
        }
        if (!empty($tmp['parameter'])) {
            $message .= 'Parameter name: ' . $tmp['parameter'] . '. ';
        }
        self::assertEquals(trim($message), trim($instance->getMessage()));

        if (empty($tmp['retry_after'])) {
            self::assertNull($instance->retryAfter);
            self::assertNull($instance->getError()->getRetryAfter());
        } else {
            self::assertEquals($tmp['retry_after'], $instance->retryAfter);
            self::assertEquals($tmp['retry_after'], $instance->getError()->getRetryAfter());
        }
        if (empty($tmp['type'])) {
            self::assertNull($instance->type);
            self::assertNull($instance->getError()->getType());
        } else {
            self::assertEquals($tmp['type'], $instance->type);
            self::assertEquals($tmp['type'], $instance->getError()->getType());
        }
    }

    public static function messageDataProvider(): array
    {
        return [
            ['{}'],
            ['{"code":"server_error","description":"Internal server error"}'],
            ['{"code":"server_error","description":"Invalid parameter value","parameter":"shop_id"}'],
            ['{"code":"server_error","description":"Invalid parameter value","parameter":"shop_id","type":"test"}'],
            ['{"code":"server_error","description":"Invalid parameter value","parameter":"shop_id","retry_after":333}'],
        ];
    }

    abstract public function expectedHttpCode();

    public function testExceptionCode(): void
    {
        $instance = $this->getTestInstance();
        self::assertEquals($this->expectedHttpCode(), $instance->getCode());
    }

    public function testConstruct(): void
    {
        $message = 'Test message';
        $code = $this->expectedHttpCode();
        $responseHeaders = ['Content-Type' => 'application/json'];
        $responseBody = '{"type": "error", "id": "test_id", "code": "unknown", "description": "Test description", "parameter": "test_parameter"}';

        $exception = $this->getTestInstance($message, $code, $responseHeaders, $responseBody);

        $this->assertEquals($code, $exception->getCode());
        $this->assertEquals($responseHeaders, $exception->getResponseHeaders());
        $this->assertEquals($responseBody, $exception->getResponseBody());

        $this->assertEquals('test_id', $exception->getErrorId());
        $this->assertEquals('test_id', $exception->getError()->getId());
        $this->assertEquals('Test description. Error code: unknown. Parameter name: test_parameter.', $exception->getMessage());
        $this->assertEquals('unknown', $exception->getErrorCode());
        $this->assertEquals('unknown', $exception->getError()->getCode());
        $this->assertEquals('Test description', $exception->getErrorDescription());
        $this->assertEquals('Test description', $exception->getError()->getDescription());
        $this->assertEquals('test_parameter', $exception->getErrorParameter());
        $this->assertEquals('test_parameter', $exception->getError()->getParameter());
    }
}
