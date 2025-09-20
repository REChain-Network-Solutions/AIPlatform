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

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\SettlementPayoutPaymentType;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\SettlementType;
use YooKassa\Request\Payments\CreateCaptureRequest;
use YooKassa\Request\Payments\CreateCaptureRequestSerializer;
use YooKassa\Request\Payments\TransferData;

/**
 * CreateCaptureRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateCaptureRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     */
    public function testSerialize(array $options): void
    {
        $serializer = new CreateCaptureRequestSerializer();
        $data = $serializer->serialize(CreateCaptureRequest::builder()->build($options));

        $expected = [];
        if (isset($options['amount'])) {
            $expected = [
                'amount' => $options['amount'],
            ];
        }
        if (!empty($options['receiptItems'])) {
            foreach ($options['receiptItems'] as $item) {
                $itemArray = [
                    'description' => $item['description'],
                    'quantity' => $item['quantity'],
                    'amount' => $item['price'],
                    'vat_code' => $item['vatCode'],
                ];

                if (!empty($item['payment_subject'])) {
                    $itemArray['payment_subject'] = $item['payment_subject'];
                }
                if (!empty($item['payment_mode'])) {
                    $itemArray['payment_mode'] = $item['payment_mode'];
                }
                if (!empty($item['product_code'])) {
                    $itemArray['product_code'] = $item['product_code'];
                }
                if (!empty($item['country_of_origin_code'])) {
                    $itemArray['country_of_origin_code'] = $item['country_of_origin_code'];
                }
                if (!empty($item['customs_declaration_number'])) {
                    $itemArray['customs_declaration_number'] = $item['customs_declaration_number'];
                }
                if (!empty($item['excise'])) {
                    $itemArray['excise'] = $item['excise'];
                }
                if (!empty($item['payment_subject_industry_details'])) {
                    $itemArray['payment_subject_industry_details'] = $item['payment_subject_industry_details'];
                }
                $expected['receipt']['items'][] = $itemArray;
            }
            if (!empty($options['receiptEmail'])) {
                $expected['receipt']['customer']['email'] = $options['receiptEmail'];
            }
            if (!empty($options['receiptEmail'])) {
                $expected['receipt']['customer']['email'] = $options['receiptEmail'];
            }
            if (!empty($options['taxSystemCode'])) {
                $expected['receipt']['tax_system_code'] = $options['taxSystemCode'];
            }
        } elseif (!empty($options['receipt'])) {
            $expected['receipt'] = $options['receipt'];
            if (!empty($expected['receipt']['phone'])) {
                $expected['receipt']['customer']['phone'] = $expected['receipt']['phone'];
                unset($expected['receipt']['phone']);
            }
            if (!empty($expected['receipt']['email'])) {
                $expected['receipt']['customer']['email'] = $expected['receipt']['email'];
                unset($expected['receipt']['email']);
            }
        }
        if (isset($options['deal'])) {
            $expected['deal'] = $options['deal'];
        }
        if (!empty($options['transfers'])) {
            foreach ($options['transfers'] as $transfer) {
                $transferData['account_id'] = $transfer['account_id'];
                if (!empty($transfer['amount'])) {
                    $transferData['amount'] = [
                        'value' => $transfer['amount']['value'],
                        'currency' => $transfer['amount']['currency'] ?? CurrencyCode::RUB,
                    ];
                }
                if (!empty($transfer['platform_fee_amount'])) {
                    $transferData['platform_fee_amount'] = [
                        'value' => $transfer['platform_fee_amount']['value'],
                        'currency' => isset($transfer['platform_fee_amount']['currency'])
                            ? $transfer['amount']['currency']
                            : CurrencyCode::RUB,
                    ];
                }
                if (!empty($transfer['description'])) {
                    $transferData['description'] = $transfer['description'];
                }
                $expected['transfers'][] = $transferData;
            }
        }
        self::assertEquals($expected, $data);
    }

    public static function validDataProvider(): array
    {
        $result = [
            [
                [],
            ],
            [
                [
                    'receiptItems' => [
                        [
                            'description' => Random::str(10),
                            'quantity' => round(Random::float(0.01, 10.00), 2),
                            'price' => [
                                'value' => round(Random::float(10.00, 100.00), 2),
                                'currency' => Random::value(CurrencyCode::getValidValues()),
                            ],
                            'vatCode' => Random::int(1, 6),
                            'payment_mode' => Random::value(PaymentMode::getValidValues()),
                            'payment_subject' => Random::value(PaymentSubject::getValidValues()),
                            'product_code' => Random::str(96, 96, '0123456789ABCDEF '),
                            'country_of_origin_code' => 'RU',
                            'customs_declaration_number' => Random::str(32),
                            'excise' => Random::float(0.0, 99.99),
                            'payment_subject_industry_details' => [
                                [
                                    'federal_id' => '001',
                                    'document_date' => date('Y-m-d', Random::int(100000000, 200000000)),
                                    'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                                    'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                                ]
                            ]
                        ],
                    ],
                    'receiptEmail' => 'johndoe@yoomoney.ru',
                    'taxSystemCode' => Random::int(1, 6),
                    'transfers' => [
                        new TransferData([
                            'account_id' => Random::str(36),
                            'amount' => new MonetaryAmount(Random::int(1, 1000), 'RUB'),
                            'platform_fee_amount' => new MonetaryAmount(Random::int(1, 1000), 'RUB'),
                            'description' => Random::str(1, TransferData::MAX_LENGTH_DESCRIPTION),
                        ]),
                    ],
                    'deal' => [
                        'id' => Random::str(36, 50),
                        'settlements' => [
                            [
                                'type' => SettlementPayoutPaymentType::PAYOUT,
                                'amount' => [
                                    'value' => round(Random::float(10.00, 100.00), 2),
                                    'currency' => Random::value(CurrencyCode::getValidValues()),
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            [
                [
                    'receipt' => [
                        'items' => [
                            [
                                'description' => Random::str(10),
                                'quantity' => round(Random::float(0.01, 10.00), 2),
                                'amount' => [
                                    'value' => round(Random::float(10.00, 100.00), 2),
                                    'currency' => Random::value(CurrencyCode::getValidValues()),
                                ],
                                'vat_code' => Random::int(1, 6),
                                'payment_subject_industry_details' => [
                                    [
                                        'federal_id' => '001',
                                        'document_date' => date('Y-m-d', Random::int(100000000, 200000000)),
                                        'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                                        'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                                    ]
                                ]
                            ],
                            [
                                'description' => Random::str(10),
                                'amount' => [
                                    'value' => round(Random::float(10.00, 100.00), 2),
                                    'currency' => Random::value(CurrencyCode::getValidValues()),
                                ],
                                'quantity' => round(Random::float(0.01, 10.00), 2),
                                'vat_code' => Random::int(1, 6),
                                'payment_subject_industry_details' => [
                                    [
                                        'federal_id' => '001',
                                        'document_date' => date('Y-m-d', Random::int(100000000, 200000000)),
                                        'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                                        'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                                    ]
                                ]
                            ],
                        ],
                        'settlements' => [
                            [
                                'type' => Random::value(SettlementType::getValidValues()),
                                'amount' => [
                                    'value' => sprintf('%.2f', round(Random::float(0.1, 99.99), 2)),
                                    'currency' => Random::value(CurrencyCode::getValidValues()),
                                ],
                            ]
                        ],
                        'customer' => [
                            'phone' => Random::str(12, '0123456789'),
                            'email' => 'johndoe@yoomoney.ru',
                            'full_name' => Random::str(1, 256),
                            'inn' => Random::str(12, 12, '1234567890'),
                        ],
                        'tax_system_code' => Random::int(1, 6),
                    ],
                    'deal' => [
                        'id' => Random::str(36, 50),
                        'settlements' => [
                            [
                                'type' => Random::value(SettlementPayoutPaymentType::getValidValues()),
                                'amount' => [
                                    'value' => sprintf('%.2f', round(Random::float(0.1, 99.99), 2)),
                                    'currency' => Random::value(CurrencyCode::getValidValues()),
                                ],
                            ]
                        ],
                    ],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'amount' => [
                    'value' => (float) Random::int(1, 1000000),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
            ];
            $result[] = [$request];
        }

        return $result;
    }
}
