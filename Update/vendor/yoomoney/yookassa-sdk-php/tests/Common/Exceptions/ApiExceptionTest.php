<?php

namespace Tests\YooKassa\Common\Exceptions;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\Exceptions\ApiException;

/**
 * @internal
 */
class ApiExceptionTest extends TestCase
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new ApiException($message, $code, $responseHeaders, $responseBody);
    }

    /**
     * @dataProvider responseHeadersDataProvider
     */
    public function testGetResponseHeaders(array $headers): void
    {
        $instance = $this->getTestInstance('', 0, $headers);
        self::assertEquals($headers, $instance->getResponseHeaders());
    }

    public static function responseHeadersDataProvider()
    {
        return [
            [
                [],
            ],
            [
                ['HTTP/1.1 200 Ok'],
            ],
            [
                [
                    'HTTP/1.1 200 Ok',
                    'Content-length: 0',
                ],
            ],
            [
                [
                    'HTTP/1.1 200 Ok',
                    'Content-length: 0',
                    'Connection: close',
                ],
            ],
        ];
    }

    /**
     * @dataProvider responseBodyDataProvider
     */
    public function testGetResponseBody(string $body): void
    {
        $instance = $this->getTestInstance('', 0, [], $body);
        self::assertEquals($body, $instance->getResponseBody());
    }

    public static function responseBodyDataProvider()
    {
        return [
            [
                '',
            ],
            [
                '{"success":true}',
            ],
            [
                '<html></html>',
            ],
        ];
    }
}
