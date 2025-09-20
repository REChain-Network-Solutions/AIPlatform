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

namespace Tests\YooKassa\Request\Receipts;

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\ReceiptRegistrationStatus;
use YooKassa\Model\Refund\RefundStatus;
use YooKassa\Request\Receipts\ReceiptsRequest;
use YooKassa\Request\Receipts\ReceiptsRequestSerializer;

/**
 * ReceiptsRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptsRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $value
     */
    public function testSerialize(mixed $value): void
    {
        $serializer = new ReceiptsRequestSerializer();
        $instance = ReceiptsRequest::builder()->build($value);
        $data = $serializer->serialize($instance);

        $expected = [];

        if (!empty($value)) {
            $expected = [
                'payment_id' => $value['paymentId'],
                'refund_id' => $value['refundId'],
                'status' => $value['status'],
                'limit' => $value['limit'],
                'cursor' => $value['cursor'],
            ];

            if (!empty($value['createdAtLt'])) {
                $expected['created_at.lt'] = $value['createdAtLt'];
            }

            if (!empty($value['createdAtGt'])) {
                $expected['created_at.gt'] = $value['createdAtGt'];
            }

            if (!empty($value['createdAtLte'])) {
                $expected['created_at.lte'] = $value['createdAtLte'];
            }

            if (!empty($value['createdAtGte'])) {
                $expected['created_at.gte'] = $value['createdAtGte'];
            }
        }

        self::assertEquals($expected, $data);
    }

    public static function validDataProvider(): array
    {
        $result = [
            [
                [
                    'paymentId' => '216749da-000f-50be-b000-096747fad91e',
                    'refundId' => '216749f7-0016-50be-b000-078d43a63ae4',
                    'status' => RefundStatus::SUCCEEDED,
                    'limit' => 100,
                    'cursor' => '37a5c87d-3984-51e8-a7f3-8de646d39ec15',
                    'createdAtGte' => date(YOOKASSA_DATE, Random::int(1, time())),
                    'createdAtGt' => date(YOOKASSA_DATE, Random::int(1, time())),
                    'createdAtLte' => date(YOOKASSA_DATE, Random::int(1, time())),
                    'createdAtLt' => date(YOOKASSA_DATE, Random::int(1, time())),
                ],
            ],
            [
                [],
            ],
        ];
        for ($i = 0; $i < 8; $i++) {
            $receipts = [
                'paymentId' => Random::str(36),
                'refundId' => Random::str(36),
                'createdAtGte' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtGt' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtLte' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtLt' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'status' => Random::value(ReceiptRegistrationStatus::getValidValues()),
                'cursor' => uniqid('', true),
                'limit' => Random::int(1, ReceiptsRequest::MAX_LIMIT_VALUE),
            ];
            $result[] = [$receipts];
        }

        return $result;
    }
}
