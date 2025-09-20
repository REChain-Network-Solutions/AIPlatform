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

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\SettlementType;
use YooKassa\Request\Receipts\CreatePostReceiptRequest;
use YooKassa\Request\Receipts\CreatePostReceiptRequestSerializer;

/**
 * CreatePostReceiptRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePostReceiptRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new CreatePostReceiptRequestSerializer();
        $instance = CreatePostReceiptRequest::builder()->build($options);
        $data = $serializer->serialize($instance);

        $expected = [
            'type' => $options['type'],
            'send' => $options['send'],
        ];

        if (!empty($options['customer'])) {
            $expected['customer'] = $options['customer'];
        }
        if (!empty($options['tax_system_code'])) {
            $expected['tax_system_code'] = $options['tax_system_code'];
        }
        if (!empty($options['items'])) {
            foreach ($options['items'] as $item) {
                $itemArray = $item;

                if (!empty($item['payment_subject'])) {
                    $itemArray['payment_subject'] = $item['payment_subject'];
                }
                if (!empty($item['payment_mode'])) {
                    $itemArray['payment_mode'] = $item['payment_mode'];
                }
                if (!empty($item['vat_code'])) {
                    $itemArray['vat_code'] = $item['vat_code'];
                }
                if (!empty($item['product_code'])) {
                    $itemArray['product_code'] = $item['product_code'];
                }
                $expected['items'][] = $itemArray;
            }
        }

        if (!empty($options['settlements'])) {
            foreach ($options['settlements'] as $item) {
                $itemArray = $item;
                $expected['settlements'][] = $itemArray;
            }
        }

        if (!empty($options['payment_id'])) {
            $expected['payment_id'] = $options['payment_id'];
        }
        if (!empty($options['refund_id'])) {
            $expected['refund_id'] = $options['refund_id'];
        }
        if (!empty($options['receipt_industry_details'])) {
            $expected['receipt_industry_details'] = $options['receipt_industry_details'];
        }

        self::assertEquals($expected, $data);
    }

    public function validDataProvider(): array
    {
        $result = [
            [
                [
                    'type' => 'payment',
                    'send' => true,
                    'customer' => [
                        'email' => 'johndoe@yoomoney.ru',
                    ],
                    'items' => [
                        [
                            'description' => Random::str(10),
                            'quantity' => (float) Random::int(1, 10),
                            'amount' => [
                                'value' => round(Random::float(1, 100), 2),
                                'currency' => CurrencyCode::RUB,
                            ],
                            'vat_code' => Random::int(1, 6),
                            'payment_subject' => PaymentSubject::COMMODITY,
                            'payment_mode' => PaymentMode::PARTIAL_PREPAYMENT,
                        ],
                    ],
                    'settlements' => [
                        [
                            'type' => Random::value(SettlementType::getValidValues()),
                            'amount' => [
                                'value' => round(Random::float(0.1, 99.99), 2),
                                'currency' => Random::value(CurrencyCode::getValidValues()),
                            ],
                        ],
                    ],
                    'payment_id' => uniqid('', true),
                    'tax_system_code' => Random::int(1, 6),
                ],
            ],
        ];

        for ($i = 0; $i < 10; $i++) {
            $type = Random::value([ReceiptType::PAYMENT, ReceiptType::REFUND]);
            $request = [
                'items' => $this->getReceiptItems($i + 1),
                'customer' => [
                    'email' => 'johndoe@yoomoney.ru',
                    'phone' => Random::str(12, '0123456789'),
                ],
                'tax_system_code' => Random::int(1, 6),
                'type' => $type,
                'send' => true,
                'settlements' => $this->getSettlements($i + 1),
                'receipt_industry_details' => [],
                $type . '_id' => uniqid('', true),
            ];
            $result[] = [$request];
        }

        return $result;
    }

    /**
     * @throws Exception
     */
    private function getReceiptItems(int $count): array
    {
        $result = [];
        for ($i = 0; $i < $count; $i++) {
            $result[] = [
                'description' => Random::str(10),
                'quantity' => (float) Random::float(1, 100),
                'amount' => [
                    'value' => (float) Random::int(1, 100),
                    'currency' => CurrencyCode::RUB,
                ],
                'vat_code' => Random::int(1, 6),
                'payment_subject' => Random::value(PaymentSubject::getValidValues()),
                'payment_mode' => Random::value(PaymentMode::getValidValues()),
                'product_code' => Random::str(96, 96, '0123456789ABCDEF '),
                'country_of_origin_code' => 'RU',
                'customs_declaration_number' => Random::str(32),
                'excise' => Random::float(0.0, 99.99),
            ];
        }

        return $result;
    }

    /**
     * @throws Exception
     */
    private function getSettlements(int $count): array
    {
        $result = [];
        for ($i = 0; $i < $count; $i++) {
            $result[] = [
                'type' => Random::value(SettlementType::getValidValues()),
                'amount' => [
                    'value' => round(Random::float(0.1, 99.99), 2),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
            ];
        }

        return $result;
    }
}
