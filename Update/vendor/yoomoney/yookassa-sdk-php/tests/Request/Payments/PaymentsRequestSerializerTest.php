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

use DateTime;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payment\PaymentStatus;
use YooKassa\Request\Payments\PaymentsRequest;
use YooKassa\Request\Payments\PaymentsRequestSerializer;

/**
 * PaymentsRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentsRequestSerializerTest extends TestCase
{
    private array $fieldMap = [
        'createdAtGte' => 'created_at.gte',
        'createdAtGt' => 'created_at.gt',
        'createdAtLte' => 'created_at.lte',
        'createdAtLt' => 'created_at.lt',
        'capturedAtGte' => 'captured_at.gte',
        'capturedAtGt' => 'captured_at.gt',
        'capturedAtLte' => 'captured_at.lte',
        'capturedAtLt' => 'captured_at.lt',
        'status' => 'status',
        'paymentMethod' => 'payment_method',
        'limit' => 'limit',
        'cursor' => 'cursor',
    ];

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new PaymentsRequestSerializer();
        $data = $serializer->serialize(PaymentsRequest::builder()->build($options));

        $expected = [];
        foreach ($this->fieldMap as $field => $mapped) {
            if (isset($options[$field])) {
                $value = $options[$field];
                if (!empty($value)) {
                    $expected[$mapped] = $value instanceof DateTime ? $value->format(YOOKASSA_DATE) : $value;
                }
            }
        }
        self::assertEquals($expected, $data);
    }

    public function validDataProvider(): array
    {
        $result = [
            [
                [],
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
                'paymentMethod' => $methods[Random::int(0, count($methods) - 1)],
                'status' => $statuses[Random::int(0, count($statuses) - 1)],
                'limit' => Random::int(1, 100),
                'cursor' => Random::str(Random::int(2, 30)),
            ];
            $result[] = [$request];
        }

        return $result;
    }

}
