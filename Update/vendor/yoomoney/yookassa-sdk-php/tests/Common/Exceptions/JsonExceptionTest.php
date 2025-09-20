<?php

namespace Tests\YooKassa\Common\Exceptions;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\Exceptions\JsonException;

/**
 * @internal
 */
class JsonExceptionTest extends TestCase
{
    /**
     * @dataProvider messageDataProvider
     *
     * @param mixed $message
     * @param mixed $code
     */
    public function testGetMessage($message, $code): void
    {
        $instance = new JsonException($message, $code);

        if (array_key_exists($code, JsonException::$errorLabels)) {
            $message .= ' ' . JsonException::$errorLabels[$code];
        } else {
            $message .= ' Unknown error';
        }
        self::assertEquals($message, $instance->getMessage());
    }

    public static function messageDataProvider()
    {
        $result = [];
        foreach (JsonException::$errorLabels as $code => $message) {
            $result[] = [$message, $code];
        }
        $result[] = ['Test error', -1];

        return $result;
    }
}
