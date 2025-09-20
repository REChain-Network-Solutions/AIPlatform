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

namespace Tests\YooKassa\Request\Refunds;

use DateTime;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Refund\RefundStatus;
use YooKassa\Request\Refunds\RefundsRequest;
use YooKassa\Request\Refunds\RefundsRequestSerializer;

/**
 * RefundsRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RefundsRequestSerializerTest extends TestCase
{
    private array $fieldMap = [
        'paymentId' => 'payment_id',
        'createdAtGte' => 'created_at.gte',
        'createdAtGt' => 'created_at.gt',
        'createdAtLte' => 'created_at.lte',
        'createdAtLt' => 'created_at.lt',
        'status' => 'status',
        'cursor' => 'cursor',
        'limit' => 'limit',
    ];

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new RefundsRequestSerializer();
        $data = $serializer->serialize(RefundsRequest::builder()->build($options));

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
                [
                    'accountId' => uniqid('', true),
                ],
            ],
            [
                [
                    'paymentId' => '',
                    'createdAtGte' => '',
                    'createdAtGt' => '',
                    'createdAtLte' => '',
                    'createdAtLt' => '',
                    'status' => '',
                    'cursor' => '',
                    'limit' => 10,
                ],
            ],
        ];
        $statuses = RefundStatus::getValidValues();
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'paymentId' => $this->randomString(),
                'createdAtGte' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtGt' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtLte' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'createdAtLt' => (0 === $i ? null : (1 === $i ? '' : date(YOOKASSA_DATE, Random::int(1, time())))),
                'status' => $statuses[Random::int(0, count($statuses) - 1)],
                'cursor' => uniqid('', true),
                'limit' => Random::int(1, RefundsRequest::MAX_LIMIT_VALUE),
            ];
            $result[] = [$request];
        }

        return $result;
    }

    private function randomString($any = true): string
    {
        static $chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-+_.';

        $result = '';
        for ($i = 0; $i < 36; $i++) {
            if ($any) {
                $char = chr(Random::int(32, 126));
            } else {
                $rnd = Random::int(0, strlen($chars) - 1);
                $char = $chars[$rnd];
            }
            $result .= $char;
        }

        return $result;
    }
}
