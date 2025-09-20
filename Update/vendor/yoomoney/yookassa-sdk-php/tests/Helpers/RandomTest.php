<?php

namespace Tests\YooKassa\Helpers;

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;

/**
 * @internal
 */
class RandomTest extends TestCase
{
    public const COUNT = 10;

    /**
     * @dataProvider randomIntDataProvider
     *
     * @throws Exception
     */
    public function testRandomInt(?int $min, ?int $max): void
    {
        $expectedMin = $min;
        if (null === $expectedMin) {
            $expectedMin = 0;
        }
        $expectedMax = $max;
        if (null === $expectedMax) {
            $expectedMax = PHP_INT_MAX;
        }
        for ($i = 0; $i < self::COUNT; $i++) {
            $value = Random::int($min, $max);
            self::assertGreaterThanOrEqual($expectedMin, $value);
            self::assertGreaterThanOrEqual($value, $expectedMax);
        }
        for ($i = 0; $i < self::COUNT; $i++) {
            $value = Random::int($min, $max);
            self::assertGreaterThanOrEqual($expectedMin, $value);
            self::assertGreaterThanOrEqual($value, $expectedMax);
        }
    }

    /**
     * @dataProvider randomFloatDataProvider
     */
    public function testRandomFloat(?float $min, ?float $max): void
    {
        $expectedMin = $min;
        if (null === $expectedMin) {
            $expectedMin = 0.0;
        }
        $expectedMax = $max;
        if (null === $expectedMax) {
            $expectedMax = 1.0;
        }
        for ($i = 0; $i < self::COUNT; $i++) {
            $value = Random::float($min, $max);
            self::assertGreaterThanOrEqual($expectedMin, $value);
            self::assertGreaterThanOrEqual($value, $expectedMax);
        }
        for ($i = 0; $i < self::COUNT; $i++) {
            $value = Random::float($min, $max);
            self::assertGreaterThanOrEqual($expectedMin, $value);
            self::assertGreaterThanOrEqual($value, $expectedMax);
        }
    }

    public function testRandomString(): void
    {
        $random = Random::str(10);
        self::assertEquals(10, strlen($random));
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            $charCode = ord($random[$i]);
            self::assertGreaterThanOrEqual(32, $charCode);
            self::assertGreaterThanOrEqual($charCode, 125);
        }

        $chars = '01';
        $random = Random::str(100, $chars);
        self::assertEquals(100, strlen($random));
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            self::assertGreaterThanOrEqual(0, strpos($random[$i], $chars));
        }

        $chars = 'abcdef';
        $random = Random::str(100, $chars);
        self::assertEquals(100, strlen($random));
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            self::assertGreaterThanOrEqual(0, strpos($random[$i], $chars));
        }

        $random = Random::str(1, 10);
        self::assertGreaterThanOrEqual(1, strlen($random));
        self::assertGreaterThanOrEqual(strlen($random), 10);
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            $charCode = ord($random[$i]);
            self::assertGreaterThanOrEqual(32, $charCode);
            self::assertGreaterThanOrEqual($charCode, 125);
        }
    }

    public function testRandomHexString(): void
    {
        $chars = '0123456789abcdef';
        $random = Random::hex(1000);
        self::assertEquals(1000, strlen($random));
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            self::assertGreaterThanOrEqual(0, strpos($random[$i], $chars));
        }

        $random = Random::hex(1000, false);
        self::assertEquals(1000, strlen($random));
        for ($i = 0, $iMax = strlen($random); $i < $iMax; $i++) {
            self::assertGreaterThanOrEqual(0, strpos($random[$i], $chars));
        }
    }

    public function testRandomBytes(): void
    {
        $random = Random::bytes(10);
        self::assertEquals(10, strlen($random));

        $random = Random::bytes(10);
        self::assertEquals(10, strlen($random));
    }

    public function testRandomValues(): void
    {
        $values = ['one', 'two', 'three'];
        $value = Random::value($values);
        self::assertContains($value, $values);
        $value = Random::value($values);
        self::assertContains($value, $values);
        $value = Random::value($values);
        self::assertContains($value, $values);

        $values = ['one'];
        $value = Random::value($values);
        self::assertContains($value, $values);
        $value = Random::value($values);
        self::assertContains($value, $values);
        $value = Random::value($values);
        self::assertContains($value, $values);
    }

    public static function randomIntDataProvider(): array
    {
        $result = [];
        $result[] = [null, null];
        $result[] = [null, 1];
        $result[] = [0, null];
        for ($i = 0; $i < self::COUNT; $i++) {
            $min = $i;
            $max = $i + Random::int(-100, 100);
            if ($min < $max) {
                $result[] = [$min, $max];
            } else {
                $result[] = [$max, $min];
            }
        }

        return $result;
    }

    public static function randomFloatDataProvider(): array
    {
        $result = [];
        $result[] = [null, null];
        $result[] = [null, 1];
        $result[] = [0, null];
        for ($i = 0; $i < self::COUNT; $i++) {
            $min = $i / 3.1415;
            $max = $i + Random::int(-100000, 1000000) / 3.141592;
            if ($min < $max) {
                $result[] = [$min, $max];
            } else {
                $result[] = [$max, $min];
            }
        }

        return $result;
    }

    public function testRandomBool(): void
    {
        for ($i = 0; $i < 10; $i++) {
            $value = Random::bool();
            self::assertIsBool($value);
            if ($value) {
                self::assertTrue($value);
            } else {
                self::assertFalse($value);
            }
        }
    }
}
