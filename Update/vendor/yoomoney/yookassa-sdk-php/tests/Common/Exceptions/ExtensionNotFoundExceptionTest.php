<?php

namespace Tests\YooKassa\Common\Exceptions;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\Exceptions\ExtensionNotFoundException;

/**
 * @internal
 */
class ExtensionNotFoundExceptionTest extends TestCase
{
    /**
     * @dataProvider messageDataProvider
     *
     * @param mixed $name
     * @param mixed $excepted
     */
    public function testGetMessage($name, $excepted): void
    {
        $instance = $this->getTestInstance($name);

        self::assertEquals($excepted, $instance->getMessage());
    }

    public static function messageDataProvider()
    {
        return [
            ['json', 'json extension is not loaded!'],
            ['curl', 'curl extension is not loaded!'],
            ['gd', 'gd extension is not loaded!'],
        ];
    }

    protected function getTestInstance(string $name, int $code = 0): ExtensionNotFoundException
    {
        return new ExtensionNotFoundException($name, $code);
    }
}
