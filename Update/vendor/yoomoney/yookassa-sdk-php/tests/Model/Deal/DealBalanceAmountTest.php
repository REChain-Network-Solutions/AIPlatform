<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace Tests\YooKassa\Model\Deal;

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\DealBalanceAmount;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * DealBalanceAmountTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class DealBalanceAmountTest extends TestCase
{
    public const DEFAULT_CURRENCY = CurrencyCode::RUB;
    public const DEFAULT_VALUE = '0.00';

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $value
     * @param mixed $currency
     */
    public function testConstructor($value, $currency): void
    {
        $instance = new DealBalanceAmount($value, $currency);

        self::assertEquals(number_format($value, 2, '.', ''), $instance->getValue());
        self::assertEquals(strtoupper($currency), $instance->getCurrency());
        self::assertNotNull($instance->getIntegerValue());
    }

    /**
     * @dataProvider validArrayDataProvider
     *
     * @param mixed $data
     */
    public function testArrayConstructor($data): void
    {
        $instance = new DealBalanceAmount();

        self::assertEquals(self::DEFAULT_VALUE, $instance->getValue());
        self::assertEquals(self::DEFAULT_CURRENCY, $instance->getCurrency());

        $instance = new DealBalanceAmount($data);

        self::assertEquals(number_format($data['value'], 2, '.', ''), $instance->getValue());
        self::assertEquals(strtoupper($data['currency']), $instance->getCurrency());
    }

    /**
     * @dataProvider validValueDataProvider
     *
     * @param mixed $value
     */
    public function testGetSetValue($value): void
    {
        $expected = number_format($value, 2, '.', '');

        $instance = self::getInstance();
        self::assertEquals(self::DEFAULT_VALUE, $instance->getValue());
        self::assertEquals(self::DEFAULT_VALUE, $instance->value);
        $instance->setValue($value);
        self::assertEquals($expected, $instance->getValue());
        self::assertEquals($expected, $instance->value);

        $instance = self::getInstance();
        $instance->value = $value;
        self::assertEquals($expected, $instance->getValue());
        self::assertEquals($expected, $instance->value);
    }

    /**
     * @dataProvider invalidValueDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidValue($value, string $exceptionClassName): void
    {
        $instance = self::getInstance();

        $this->expectException($exceptionClassName);
        $instance->setValue($value);
    }

    /**
     * @dataProvider invalidValueDataProvider
     *
     * @param mixed $value
     */
    public function testSetterInvalidValue($value, string $exceptionClassName): void
    {
        $instance = self::getInstance();

        $this->expectException($exceptionClassName);
        $instance->value = $value;
    }

    /**
     * @dataProvider validCurrencyDataProvider
     */
    public function testGetSetCurrency(string $currency): void
    {
        $instance = self::getInstance();

        self::assertEquals(self::DEFAULT_CURRENCY, $instance->getCurrency());
        self::assertEquals(self::DEFAULT_CURRENCY, $instance->currency);
        $instance->setCurrency($currency);
        self::assertEquals(strtoupper($currency), $instance->getCurrency());
        self::assertEquals(strtoupper($currency), $instance->currency);
    }

    /**
     * @dataProvider invalidCurrencyDataProvider
     */
    public function testSetInvalidCurrency(mixed $currency, string $exceptionClassName): void
    {
        $instance = self::getInstance();

        $this->expectException($exceptionClassName);
        $instance->setCurrency($currency);
    }

    /**
     * @dataProvider invalidCurrencyDataProvider
     */
    public function testSetterInvalidCurrency(mixed $currency, string $exceptionClassName): void
    {
        $instance = self::getInstance();

        $this->expectException($exceptionClassName);
        $instance->currency = $currency;
    }

    public function validDataProvider(): array
    {
        $result = $this->validValueDataProvider();
        foreach ($this->validCurrencyDataProvider() as $index => $tmp) {
            if (isset($result[$index])) {
                $result[$index][] = $tmp[0];
            }
        }

        return $result;
    }

    public static function validArrayDataProvider(): array
    {
        $result = [];
        foreach (range(1, 10) as $i) {
            $result[$i][] = [
                'value' => Random::value(['-', '']) . Random::float(0, 9999.99),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ];
        }

        return $result;
    }

    public static function validValueDataProvider(): array
    {
        return [
            [0.01],
            [0.1],
            [0.11],
            [0.1111],
            [0.1166],
            ['0.01'],
            [1],
            [0],
            [-1],
            ['100'],
            ['-100'],
        ];
    }

    public static function validCurrencyDataProvider(): array
    {
        $result = [];
        foreach (CurrencyCode::getValidValues() as $value) {
            $result[] = [$value];
        }

        return $result;
    }

    public static function invalidValueDataProvider(): array
    {
        return [
            [null, EmptyPropertyValueException::class],
            ['', EmptyPropertyValueException::class],
            [[], EmptyPropertyValueException::class],
            [fopen(__FILE__, 'r'), InvalidPropertyValueTypeException::class],
            ['invalid_value', InvalidPropertyValueTypeException::class],
            [true, InvalidPropertyValueTypeException::class],
            [false, EmptyPropertyValueException::class],
        ];
    }

    public static function invalidCurrencyDataProvider(): array
    {
        return [
            [null, EmptyPropertyValueException::class],
            ['invalid_value', InvalidPropertyValueException::class],
            ['III', InvalidPropertyValueException::class],
        ];
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $value
     * @param mixed $currency
     */
    public function testJsonSerialize($value, $currency): void
    {
        $instance = new DealBalanceAmount($value, $currency);
        $expected = [
            'value' => number_format($value, 2, '.', ''),
            'currency' => strtoupper($currency),
        ];
        self::assertEquals($expected, $instance->jsonSerialize());
    }

    protected static function getInstance($value = null, $currency = null): DealBalanceAmount
    {
        return new DealBalanceAmount($value, $currency);
    }
}
