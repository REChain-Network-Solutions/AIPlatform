<?php

namespace Tests\YooKassa\Helpers;

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\RawHeadersParser;

/**
 * @internal
 */
class RawHeadersParserTest extends TestCase
{
    /**
     * @dataProvider headersDataProvider
     *
     * @param mixed $rawHeaders
     * @param mixed $expected
     */
    public function testParse(mixed $rawHeaders, mixed $expected): void
    {
        $this->assertEquals($expected, RawHeadersParser::parse($rawHeaders));
    }

    public static function headersDataProvider(): array
    {
        return [
            [
                'Server: nginx
                Date: Thu, 03 Dec 2020 16:04:35 GMT
                Content-Type: text/html
                Content-Length: 178
                Connection: keep-alive
                Location: https://yoomoney.ru/',
                [
                    'Server' => 'nginx',
                    'Date' => 'Thu, 03 Dec 2020 16:04:35 GMT',
                    'Content-Type' => 'text/html',
                    'Content-Length' => '178',
                    'Connection' => 'keep-alive',
                    'Location' => 'https://yoomoney.ru/',
                ],
            ],
            [
                "HTTP/1.1 200 Ok\r\n" .
                "Server: nginx\r\n" .
                "Date: Thu, 03 Dec 2020 16:04:35 GMT\r\n" .
                "Content-Type: text/html\r\n" .
                "Content-Length: 178\r\n" .
                "Array-Header: value1\r\n" .
                "Connection: keep-alive\r\n" .
                "Array-Header: value2\r\n" .
                "Location: https://yoomoney.ru/\r\n" .
                "\r\n" .
                'Content-Length: 132',
                [
                    0 => 'HTTP/1.1 200 Ok',
                    'Server' => 'nginx',
                    'Date' => 'Thu, 03 Dec 2020 16:04:35 GMT',
                    'Content-Type' => 'text/html',
                    'Content-Length' => '178',
                    'Array-Header' => [
                        'value1', 'value2',
                    ],
                    'Connection' => 'keep-alive',
                    'Location' => 'https://yoomoney.ru/',
                ],
            ],
            [
                "HTTP/1.1 200 Ok\r\n" .
                "Server: nginx\r\n" .
                "\tversion 1.3.4\r\n" .
                "Date: Thu, 03 Dec 2020 16:04:35 GMT\r\n" .
                "Content-Type: text/html\r\n" .
                "Content-Length: 178\r\n" .
                "Array-Header: value1\r\n" .
                "Connection: keep-alive\r\n" .
                "Array-Header: value2\r\n" .
                "Location: https://yoomoney.ru/\r\n" .
                "Array-Header: value3\r\n" .
                "\r\n" .
                'Content-Length: 132',
                [
                    0 => 'HTTP/1.1 200 Ok',
                    'Server' => "nginx\r\n\tversion 1.3.4",
                    'Date' => 'Thu, 03 Dec 2020 16:04:35 GMT',
                    'Content-Type' => 'text/html',
                    'Content-Length' => '178',
                    'Array-Header' => [
                        'value1', 'value2', 'value3',
                    ],
                    'Connection' => 'keep-alive',
                    'Location' => 'https://yoomoney.ru/',
                ],
            ],
        ];
    }
}
