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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;
use YooKassa\Model\Payment\PaymentMethod\BankCardType;
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory;
use YooKassa\Model\Payment\PaymentMethodType;

/**
 * PaymentMethodFactoryTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodFactoryTest extends TestCase
{
    /**
     * @dataProvider validTypeDataProvider
     */
    public function testFactory(string $type): void
    {
        $instance = $this->getTestInstance();
        $paymentData = $instance->factory($type);
        self::assertNotNull($paymentData);
        self::assertInstanceOf(AbstractPaymentMethod::class, $paymentData);
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
        self::assertInstanceOf(AbstractPaymentMethod::class, $paymentData);

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
        self::assertInstanceOf(AbstractPaymentMethod::class, $paymentData);

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
     * @param mixed $options
     */
    public function testInvalidFactoryFromArray($options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = $this->getTestInstance();
        $instance->factoryFromArray($options);
    }

    public static function validTypeDataProvider()
    {
        $result = [];
        foreach (PaymentMethodType::getValidValues() as $value) {
            $result[] = [$value];
        }

        return $result;
    }

    public static function invalidTypeDataProvider()
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

    public static function validArrayDataProvider()
    {
        $result = [
            [
                [
                    'type' => PaymentMethodType::ALFABANK,
                    'login' => Random::str(10, 20),
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::GOOGLE_PAY,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::APPLE_PAY,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::BANK_CARD,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                    'card' => [
                        'last4' => Random::str(4, '0123456789'),
                        'first6' => Random::str(6, '0123456789'),
                        'expiry_year' => Random::int(2000, 2200),
                        'expiry_month' => Random::value(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']),
                        'card_type' => Random::value(BankCardType::getValidValues()),
                    ],
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::BANK_CARD,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                    'card' => [
                        'last4' => Random::str(4, '0123456789'),
                        'first6' => Random::str(6, '0123456789'),
                        'expiry_year' => Random::int(2000, 2200),
                        'expiry_month' => Random::value(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']),
                        'card_type' => Random::value(BankCardType::getValidValues()),
                    ],
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::SBERBANK,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                    'card' => [
                        'last4' => Random::str(4, '0123456789'),
                        'first6' => Random::str(6, '0123456789'),
                        'expiry_year' => Random::int(2000, 2200),
                        'expiry_month' => Random::value(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']),
                        'card_type' => Random::value(BankCardType::getValidValues()),
                    ],
                    'phone' => Random::str(4, 15, '0123456789'),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::CASH,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::MOBILE_BALANCE,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                    'phone' => Random::str(4, 15, '0123456789'),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::SBERBANK,
                    'phone' => Random::str(4, 15, '0123456789'),
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::YOO_MONEY,
                    'account_number' => Random::str(31, '0123456789'),
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::YOO_MONEY,
                    'account_number' => Random::str(31, '0123456789'),
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::INSTALLMENTS,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::B2B_SBERBANK,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::TINKOFF_BANK,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                ],
            ],
            [
                [
                    'type' => PaymentMethodType::SBP,
                    'id' => Random::str(2, 64),
                    'saved' => Random::bool(),
                    'title' => Random::str(10, 20),
                ],
            ],
        ];
        foreach (PaymentMethodType::getValidValues() as $value) {
            $result[] = [['type' => $value]];
        }

        return $result;
    }

    public static function invalidDataArrayDataProvider()
    {
        return [
            [[]],
        ];
    }

    protected function getTestInstance(): PaymentMethodFactory
    {
        return new PaymentMethodFactory();
    }
}
