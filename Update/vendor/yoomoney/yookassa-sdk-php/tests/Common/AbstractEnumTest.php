<?php

namespace Tests\YooKassa\Common;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\AbstractEnum;

/**
 * @internal
 */
class AbstractEnumTest extends TestCase
{
    /**
     * @dataProvider enumDataProvider
     *
     * @param mixed $value
     * @param mixed $exists
     */
    public function testValueExists(mixed $value, mixed $exists): void
    {
        if ($exists) {
            self::assertTrue(TestAbstractEnum::valueExists($value));
        } else {
            self::assertFalse(TestAbstractEnum::valueExists($value));
        }
    }

    public function testGetValidValues(): void
    {
        foreach (TestAbstractEnum::getValidValues() as $value) {
            self::assertTrue(TestAbstractEnum::valueExists($value));
        }
    }

    public static function enumDataProvider(): array
    {
        return [
            [TestAbstractEnum::ENUM_VALUE_1, true],
            [TestAbstractEnum::ENUM_VALUE_2, true],
            [TestAbstractEnum::ENUM_DISABLED_VALUE_1, true],
            [TestAbstractEnum::ENUM_DISABLED_VALUE_2, true],
            ['invalid_value', false],
            [0, false],
        ];
    }

    public function testGetEnabledValues(): void
    {
        $values = TestAbstractEnum::getEnabledValues();
        foreach ($values as $value) {
            self::assertTrue(TestAbstractEnum::valueExists($value));
        }
        self::assertNotContains(TestAbstractEnum::ENUM_DISABLED_VALUE_1, $values);
        self::assertNotContains(TestAbstractEnum::ENUM_DISABLED_VALUE_2, $values);
    }
}

class TestAbstractEnum extends AbstractEnum
{
    public const ENUM_VALUE_1 = 'enum_value_1';
    public const ENUM_VALUE_2 = 'enum_value_2';
    public const ENUM_DISABLED_VALUE_1 = 'enum_disabled_value_1';
    public const ENUM_DISABLED_VALUE_2 = 'enum_disabled_value_2';

    protected static array $validValues = [
        self::ENUM_VALUE_1 => true,
        self::ENUM_VALUE_2 => true,
        self::ENUM_DISABLED_VALUE_1 => false,
        self::ENUM_DISABLED_VALUE_2 => false,
    ];
}
