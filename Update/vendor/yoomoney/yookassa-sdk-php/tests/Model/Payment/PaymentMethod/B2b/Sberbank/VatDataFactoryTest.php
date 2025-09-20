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

namespace Tests\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank;

use InvalidArgumentException;
use TypeError;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataFactory;
use PHPUnit\Framework\TestCase;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataRate;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataType;
use YooKassa\Model\Payment\PaymentMethodType;

/**
 * VatDataFactoryTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class VatDataFactoryTest extends TestCase
{
    /**
     * @dataProvider validTypeDataProvider
     */
    public function testFactory(string $type): void
    {
        $instance = $this->getTestInstance();
        $paymentData = $instance->factory($type);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractVatData::class, $paymentData);
        self::assertEquals($type, $paymentData->getType());
    }

    /**
     * @dataProvider validArrayDataProvider
     */
    public function testFactoryFromArray(array $options): void
    {
        $instance = $this->getTestInstance();

        $paymentData = $instance->factoryFromArray($options);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractVatData::class, $paymentData);

        foreach ($options as $property => $value) {
            if (is_object($paymentData->{$property})) {
                self::assertEquals($paymentData->{$property}->toArray(), $value);
            } else {
                self::assertEquals($paymentData->{$property}, $value);
            }
        }

        $type = $options['type'];
        unset($options['type']);
        $paymentData = $instance->factoryFromArray($options, $type);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractVatData::class, $paymentData);

        self::assertEquals($type, $paymentData->getType());
        foreach ($options as $property => $value) {
            if (is_object($paymentData->{$property})) {
                self::assertEquals($paymentData->{$property}->toArray(), $value);
            } else {
                self::assertEquals($paymentData->{$property}, $value);
            }
        }
    }

    /**
     * @dataProvider invalidDataArrayDataProvider
     *
     * @param mixed $value
     * @param string $exceptionClassName
     */
    public function testInvalidFactoryFromArray(mixed $value, string $exceptionClassName): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClassName);
        $instance->factoryFromArray($value);
    }

    /**
     * @dataProvider invalidFactoryDataArrayDataProvider
     *
     * @param mixed $value
     * @param string $exceptionClassName
     */
    public function testInvalidFactory(mixed $value, string $exceptionClassName): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClassName);
        $instance->factory($value);
    }

    public static function validTypeDataProvider(): array
    {
        $result = [];
        foreach (VatDataType::getValidValues() as $value) {
            $result[] = [$value];
        }

        return $result;
    }

    public static function invalidTypeDataProvider(): array
    {
        return [
            [''],
            [0],
            [1],
            [-1],
            ['5'],
            [Random::str(10)],
        ];
    }

    public static function validArrayDataProvider(): array
    {
        $result = [
            [
                [
                    'type' => VatDataType::CALCULATED,
                    'rate' => Random::value(VatDataRate::getValidValues()),
                    'amount' => [
                        'value' => Random::int(0, 9999),
                        'currency' => Random::value(CurrencyCode::getValidValues()),
                    ],
                ],
            ],
            [
                [
                    'type' => VatDataType::MIXED,
                    'amount' => [
                        'value' => Random::int(0, 9999),
                        'currency' => Random::value(CurrencyCode::getValidValues()),
                    ],
                ],
            ],
            [
                [
                    'type' => VatDataType::UNTAXED,
                ],
            ],
        ];
        foreach (VatDataType::getValidValues() as $value) {
            $result[] = [['type' => $value]];
        }

        return $result;
    }

    public static function invalidDataArrayDataProvider(): array
    {
        return [
            [null, TypeError::class],
            [[], InvalidArgumentException::class],
            [new \stdClass(), TypeError::class],
        ];
    }

    public static function invalidFactoryDataArrayDataProvider(): array
    {
        return [
            [null, InvalidArgumentException::class],
            [PaymentMethodType::UNKNOWN, InvalidArgumentException::class],
            [[], TypeError::class],
            [new \stdClass(), TypeError::class],
        ];
    }

    protected function getTestInstance(): VatDataFactory
    {
        return new VatDataFactory();
    }
}
