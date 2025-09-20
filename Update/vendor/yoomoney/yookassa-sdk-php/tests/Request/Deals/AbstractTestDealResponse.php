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

namespace Tests\YooKassa\Request\Deals;

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\DealInterface;
use YooKassa\Model\Deal\DealStatus;
use YooKassa\Model\Deal\DealType;
use YooKassa\Model\Deal\FeeMoment;

/**
 * AbstractTestDealResponse
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestDealResponse extends TestCase
{
    /**
     * @dataProvider validDataProvider
     */
    public function testGetId(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['id'], $instance->getId());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetType(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['type'], $instance->getType());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetStatus(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['status'], $instance->getStatus());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetDescription(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['description'], $instance->getDescription());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetBalance(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals(number_format($options['balance']['value'], 2, '.', ''), $instance->getBalance()->getValue());
        self::assertEquals($options['balance']['currency'], $instance->getBalance()->getCurrency());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetPayoutBalance(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['payout_balance'])) {
            self::assertNull($instance->getPayoutBalance());
        } else {
            self::assertEquals(number_format($options['payout_balance']['value'], 2, '.', ''), $instance->getPayoutBalance()->getValue());
            self::assertEquals((string) $options['payout_balance']['currency'], $instance->getPayoutBalance()->getCurrency());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetCreatedAt(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['created_at'])) {
            self::assertNull($instance->getCreatedAt());
        } else {
            self::assertEquals($options['created_at'], $instance->getCreatedAt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetExpiresAt(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['expires_at'])) {
            self::assertNull($instance->getExpiresAt());
        } else {
            self::assertEquals($options['expires_at'], $instance->getExpiresAt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetFeeMoment(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['fee_moment'], $instance->getFeeMoment());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetTest(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['test'], $instance->getTest());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetMetadata(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (!empty($options['metadata'])) {
            self::assertEquals($options['metadata'], $instance->getMetadata()->toArray());
        } else {
            self::assertNull($instance->getMetadata());
        }
    }

    public static function validDataProvider(): array
    {
        $result = [];
        $statuses = DealStatus::getValidValues();
        $types = DealType::getValidValues();

        for ($i = 0; $i < 10; $i++) {
            $deal = [
                'id' => Random::str(36),
                'type' => Random::value($types),
                'status' => Random::value($statuses),
                'description' => Random::str(128),
                'balance' => [
                    'value' => Random::float(0.01, 1000000.0),
                    'currency' => Random::value(CurrencyCode::getEnabledValues()),
                ],
                'payout_balance' => [
                    'value' => Random::float(0.01, 1000000.0),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
                'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expires_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'fee_moment' => Random::value(FeeMoment::getEnabledValues()),
                'test' => (bool) ($i % 2),
                'metadata' => [
                    'value' => Random::float(0.01, 1000000.0),
                    'currency' => Random::str(1, 256),
                ],
            ];
            $result[] = [$deal];
        }

        $trueFalse = Random::bool();
        $result[] = [
            [
                'id' => Random::str(36),
                'type' => Random::value($types),
                'status' => Random::value($statuses),
                'description' => Random::str(128),
                'balance' => [
                    'value' => Random::float(0.01, 1000000.0),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
                'payout_balance' => [
                    'value' => Random::float(0.01, 1000000.0),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
                'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'expires_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'fee_moment' => Random::value(FeeMoment::getEnabledValues()),
                'test' => $trueFalse,
                'metadata' => [],
            ],
        ];

        return $result;
    }

    /**
     * @param mixed $options
     */
    abstract protected function getTestInstance($options): DealInterface;
}
