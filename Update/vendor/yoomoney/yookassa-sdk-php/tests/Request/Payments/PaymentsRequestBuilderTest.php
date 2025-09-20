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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Helpers\UUID;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payment\PaymentStatus;
use YooKassa\Request\Payments\PaymentsRequestBuilder;

/**
 * PaymentsRequestBuilderTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentsRequestBuilderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCursor($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCursor());

        $builder->setCursor($options['cursor']);
        $instance = $builder->build();
        if (empty($options['cursor'])) {
            self::assertNull($instance->getCursor());
        } else {
            self::assertEquals($options['cursor'], $instance->getCursor());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCreatedAtGte($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtGte());

        $builder->setCreatedAtGte($options['createdAtGte']);
        $instance = $builder->build();
        if (empty($options['createdAtGte'])) {
            self::assertNull($instance->getCreatedAtGte());
        } else {
            self::assertEquals($options['createdAtGte'], $instance->getCreatedAtGte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCreatedGt($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtGt());

        $builder->setCreatedAtGt($options['createdAtGt']);
        $instance = $builder->build();
        if (empty($options['createdAtGt'])) {
            self::assertNull($instance->getCreatedAtGt());
        } else {
            self::assertEquals($options['createdAtGt'], $instance->getCreatedAtGt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCreatedLte($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtLte());

        $builder->setCreatedAtLte($options['createdAtLte']);
        $instance = $builder->build();
        if (empty($options['createdAtLte'])) {
            self::assertNull($instance->getCreatedAtLte());
        } else {
            self::assertEquals($options['createdAtLte'], $instance->getCreatedAtLte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCreatedLt($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCreatedAtLt());

        $builder->setCreatedAtLt($options['createdAtLt']);
        $instance = $builder->build();
        if (empty($options['createdAtLt'])) {
            self::assertNull($instance->getCreatedAtLt());
        } else {
            self::assertEquals($options['createdAtLt'], $instance->getCreatedAtLt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCapturedAtGte($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCapturedAtGte());

        $builder->setCapturedAtGte($options['capturedAtGte']);
        $instance = $builder->build();
        if (empty($options['capturedAtGte'])) {
            self::assertNull($instance->getCapturedAtGte());
        } else {
            self::assertEquals($options['capturedAtGte'], $instance->getCapturedAtGte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCapturedGt($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCapturedAtGt());

        $builder->setCapturedAtGt($options['capturedAtGt']);
        $instance = $builder->build();
        if (empty($options['capturedAtGt'])) {
            self::assertNull($instance->getCapturedAtGt());
        } else {
            self::assertEquals($options['capturedAtGt'], $instance->getCapturedAtGt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCapturedLte($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCapturedAtLte());

        $builder->setCapturedAtLte($options['capturedAtLte']);
        $instance = $builder->build();
        if (empty($options['capturedAtLte'])) {
            self::assertNull($instance->getCapturedAtLte());
        } else {
            self::assertEquals($options['capturedAtLte'], $instance->getCapturedAtLte()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetCapturedLt($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getCapturedAtLt());

        $builder->setCapturedAtLt($options['capturedAtLt']);
        $instance = $builder->build();
        if (empty($options['capturedAtLt'])) {
            self::assertNull($instance->getCapturedAtLt());
        } else {
            self::assertEquals($options['capturedAtLt'], $instance->getCapturedAtLt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetPaymentMethod($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getPaymentMethod());

        $builder->setPaymentMethod($options['paymentMethod']);
        $instance = $builder->build();
        if (empty($options['paymentMethod'])) {
            self::assertNull($instance->getPaymentMethod());
        } else {
            self::assertEquals($options['paymentMethod'], $instance->getPaymentMethod());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetLimit($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();
        self::assertNull($instance->getLimit());

        $builder->setLimit($options['limit']);
        $instance = $builder->build();
        if (is_null($options['limit'])) {
            self::assertNull($instance->getLimit());
        } else {
            self::assertEquals($options['limit'], $instance->getLimit());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSetStatus($options): void
    {
        $builder = new PaymentsRequestBuilder();

        $instance = $builder->build();

        $builder->setStatus($options['status']);
        $instance = $builder->build();
        if (empty($options['status'])) {
            self::assertNull($instance->getStatus());
        } else {
            self::assertEquals($options['status'], $instance->getStatus());
        }
    }

    /**
     * @throws Exception
     */
    public function validDataProvider()
    {
        $result = [
            [
                [
                    'createdAtGte' => null,
                    'createdAtGt' => null,
                    'createdAtLte' => null,
                    'createdAtLt' => null,
                    'capturedAtGte' => null,
                    'capturedAtGt' => null,
                    'capturedAtLte' => null,
                    'capturedAtLt' => null,
                    'paymentMethod' => null,
                    'status' => null,
                    'limit' => null,
                    'cursor' => null,
                ],
            ],
            [
                [
                    'createdAtGte' => '',
                    'createdAtGt' => '',
                    'createdAtLte' => '',
                    'createdAtLt' => '',
                    'capturedAtGte' => '',
                    'capturedAtGt' => '',
                    'capturedAtLte' => '',
                    'capturedAtLt' => '',
                    'paymentMethod' => '',
                    'status' => '',
                    'limit' => 1,
                    'cursor' => '',
                ],
            ],
        ];
        $statuses = PaymentStatus::getValidValues();
        $methods = PaymentMethodType::getValidValues();
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'createdAtGte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtGt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtLte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'createdAtLt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'capturedAtGte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'capturedAtGt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'capturedAtLte' => date(YOOKASSA_DATE, Random::int(1, time())),
                'capturedAtLt' => date(YOOKASSA_DATE, Random::int(1, time())),
                'paymentMethod' => Random::value($methods),
                'status' => Random::value($statuses),
                'limit' => Random::int(1, 100),
                'cursor' => UUID::v4(),
            ];
            $result[] = [$request];
        }

        return $result;
    }
}
