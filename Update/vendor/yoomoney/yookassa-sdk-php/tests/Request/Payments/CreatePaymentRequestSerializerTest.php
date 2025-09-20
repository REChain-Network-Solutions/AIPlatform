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
use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\SettlementPayoutPaymentType;
use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Model\Payment\Payment;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataRate;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataType;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payment\Transfer;
use YooKassa\Model\Payment\TransferStatus;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Request\Payments\Airline;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesExternal;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesMobileApplication;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect;
use YooKassa\Request\Payments\CreatePaymentRequest;
use YooKassa\Request\Payments\CreatePaymentRequestSerializer;
use YooKassa\Request\Payments\Leg;
use YooKassa\Request\Payments\Passenger;
use YooKassa\Request\Payments\PaymentData\PaymentDataApplePay;
use YooKassa\Request\Payments\PaymentData\PaymentDataB2bSberbank;
use YooKassa\Request\Payments\PaymentData\PaymentDataBankCard;
use YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard;
use YooKassa\Request\Payments\PaymentData\PaymentDataCash;
use YooKassa\Request\Payments\PaymentData\PaymentDataGooglePay;
use YooKassa\Request\Payments\PaymentData\PaymentDataInstallments;
use YooKassa\Request\Payments\PaymentData\PaymentDataMobileBalance;
use YooKassa\Request\Payments\PaymentData\PaymentDataQiwi;
use YooKassa\Request\Payments\PaymentData\PaymentDataSberbank;
use YooKassa\Request\Payments\PaymentData\PaymentDataSbp;
use YooKassa\Request\Payments\PaymentData\PaymentDataYooMoney;

/**
 * CreatePaymentRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePaymentRequestSerializerTest extends TestCase
{
    private array $fieldMap = [
        'payment_token' => 'paymentToken',
        'payment_method_id' => 'paymentMethodId',
        'client_ip' => 'clientIp',
        'merchant_customer_id' => 'merchantCustomerId',
    ];

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new CreatePaymentRequestSerializer();
        $instance = CreatePaymentRequest::builder()->build($options);
        $data = $serializer->serialize($instance);

        $expected = [
            'amount' => $options['amount'],
        ];
        foreach ($this->fieldMap as $mapped => $field) {
            if (isset($options[$field])) {
                $value = $options[$field];
                if (!empty($value)) {
                    $expected[$mapped] = $value instanceof DateTime ? $value->format(YOOKASSA_DATE) : $value;
                }
            }
        }
        if (!empty($options['accountId']) && !empty($options['gatewayId'])) {
            $expected['recipient'] = [
                'gateway_id' => $options['gatewayId'],
            ];
        }
        if (!empty($options['confirmation'])) {
            $expected['confirmation'] = [
                'type' => $options['confirmation']->getType(),
            ];
            if ($locale = $options['confirmation']->getLocale()) {
                $expected['confirmation']['locale'] = $locale;
            }
            if (ConfirmationType::REDIRECT === $options['confirmation']->getType()) {
                $expected['confirmation']['enforce'] = $options['confirmation']->enforce;
                $expected['confirmation']['return_url'] = $options['confirmation']->returnUrl;
            }
            if (ConfirmationType::MOBILE_APPLICATION === $options['confirmation']->getType()) {
                $expected['confirmation']['return_url'] = $options['confirmation']->returnUrl;
            }
        }
        if (!empty($options['paymentMethodData'])) {
            $expected['payment_method_data'] = [
                'type' => $options['paymentMethodData']->getType(),
            ];

            switch ($options['paymentMethodData']['type']) {
                case PaymentMethodType::ALFABANK:
                    $expected['payment_method_data']['login'] = $options['paymentMethodData']->getLogin();

                    break;

                case PaymentMethodType::APPLE_PAY:
                    $expected['payment_method_data']['payment_data'] = $options['paymentMethodData']->getPaymentData();

                    break;

                case PaymentMethodType::GOOGLE_PAY:
                    $expected['payment_method_data']['payment_method_token'] = $options['paymentMethodData']->getPaymentMethodToken();
                    $expected['payment_method_data']['google_transaction_id'] = $options['paymentMethodData']->getGoogleTransactionId();

                    break;

                case PaymentMethodType::BANK_CARD:
                    $expected['payment_method_data']['card'] = [
                        'number' => $options['paymentMethodData']->getCard()->getNumber(),
                        'expiry_year' => $options['paymentMethodData']->getCard()->getExpiryYear(),
                        'expiry_month' => $options['paymentMethodData']->getCard()->getExpiryMonth(),
                        'csc' => $options['paymentMethodData']->getCard()->getCsc(),
                        'cardholder' => $options['paymentMethodData']->getCard()->getCardholder(),
                    ];

                    break;

                case PaymentMethodType::MOBILE_BALANCE:
                case PaymentMethodType::CASH:
                case PaymentMethodType::SBERBANK:
                case PaymentMethodType::QIWI:
                    $expected['payment_method_data']['phone'] = $options['paymentMethodData']->getPhone();

                    break;

                case PaymentMethodType::B2B_SBERBANK:
                    /** @var PaymentDataB2bSberbank $paymentMethodData */
                    $paymentMethodData = $options['paymentMethodData'];
                    $expected['payment_method_data']['payment_purpose'] = $paymentMethodData->getPaymentPurpose();
                    $expected['payment_method_data']['vat_data'] = $paymentMethodData->getVatData()->toArray();

                    break;
            }
        }
        if (!empty($options['metadata'])) {
            $expected['metadata'] = [];
            foreach ($options['metadata'] as $key => $value) {
                $expected['metadata'][$key] = $value;
            }
        }
        if (!empty($options['receipt']['items'])) {
            $expected['receipt'] = ['items' => []];
            foreach ($options['receipt']['items'] as $item) {
                $itemArray = $item;

                if (!empty($item['payment_subject'])) {
                    $itemArray['payment_subject'] = $item['payment_subject'];
                }
                if (!empty($item['payment_mode'])) {
                    $itemArray['payment_mode'] = $item['payment_mode'];
                }
                if (!empty($item['payment_subject_industry_details'])) {
                    $itemArray['payment_subject_industry_details'] = $item['payment_subject_industry_details'];
                }
                $expected['receipt']['items'][] = $itemArray;
            }
            if (!empty($options['receipt']['customer'])) {
                $expected['receipt']['customer'] = $options['receipt']['customer'];
            }
            if (!empty($options['receipt']['tax_system_code'])) {
                $expected['receipt']['tax_system_code'] = $options['receipt']['tax_system_code'];
            }
        }

        if (array_key_exists('capture', $options)) {
            $expected['capture'] = (bool) $options['capture'];
        }
        if (array_key_exists('savePaymentMethod', $options) && isset($options['savePaymentMethod'])) {
            $expected['save_payment_method'] = (bool) $options['savePaymentMethod'];
        }
        if (!empty($options['description'])) {
            $expected['description'] = $options['description'];
        }

        if (!empty($options['airline'])) {
            $expected['airline'] = [
                'booking_reference' => $options['airline']['booking_reference'],
                'ticket_number' => $options['airline']['ticket_number'],
                'passengers' => array_map(static function ($passenger) {
                    return [
                        'first_name' => $passenger['first_name'],
                        'last_name' => $passenger['last_name'],
                    ];
                }, $options['airline']['passengers']),
                'legs' => array_map(static function ($leg) {
                    return [
                        'departure_airport' => $leg['departure_airport'],
                        'destination_airport' => $leg['destination_airport'],
                        'departure_date' => $leg['departure_date'] instanceof DateTime ? $leg['departure_date']->format(Leg::ISO8601) : $leg['departure_date'],
                    ];
                }, $options['airline']['legs']),
            ];
        }

        if (!empty($options['transfers'])) {
            $expected['transfers'] = [];
            foreach ($options['transfers'] as $item) {
                $itemArray = $item;

                if (!empty($item['account_id'])) {
                    $itemArray['account_id'] = $item['account_id'];
                }
                if (!empty($item['amount'])) {
                    $itemArray['amount'] = $item['amount'];
                }
                if (!empty($item['status'])) {
                    $itemArray['status'] = $item['status'];
                }
                if (!empty($item['platform_fee_amount'])) {
                    $itemArray['platform_fee_amount'] = $item['platform_fee_amount'];
                }
                if (!empty($item['description'])) {
                    $itemArray['description'] = $item['description'];
                }
                if (!empty($item['metadata'])) {
                    $itemArray['metadata'] = $item['metadata'];
                }
                $expected['transfers'][] = $itemArray;
            }
        }

        if (!empty($options['deal'])) {
            $expected['deal'] = $options['deal'];
        }

        if (!empty($options['merchant_customer_id'])) {
            $expected['merchant_customer_id'] = $options['merchant_customer_id'];
        }

        self::assertEquals($expected, $data);
    }

    /**
     * @throws Exception
     */
    public function validDataProvider(): array
    {
        $airline = new Airline();
        $airline->setBookingReference(Random::str(10));
        $airline->setTicketNumber(Random::int(10));
        $leg = new Leg();
        $leg->setDepartureAirport(Random::str(3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'));
        $leg->setDestinationAirport(Random::str(3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'));
        $leg->setDepartureDate('2018-12-31');
        $airline->setLegs([$leg]);
        $passenger = new Passenger();
        $passenger->setFirstName(Random::str(10));
        $passenger->setLastName(Random::str(10));
        $airline->setPassengers([$passenger]);
        $amount = Random::float(10, 999);

        $result = [
            [
                [
                    'amount' => [
                        'value' => sprintf('%.2f', round($amount, 2)),
                        'currency' => Random::value(CurrencyCode::getValidValues()),
                    ],
                    'paymentToken' => Random::str(36),
                    'receipt' => [
                        'items' => [
                            [
                                'description' => Random::str(10),
                                'quantity' => (float) Random::int(1, 10),
                                'amount' => [
                                    'value' => (float) Random::int(100, 100),
                                    'currency' => CurrencyCode::RUB,
                                ],
                                'vat_code' => Random::int(1, 6),
                                'payment_subject' => PaymentSubject::COMMODITY,
                                'payment_mode' => PaymentMode::PARTIAL_PREPAYMENT,
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
                        'customer' => [
                            'email' => 'johndoe@yoomoney.ru',
                        ],
                        'tax_system_code' => Random::int(1, 6),
                    ],
                    'description' => Random::str(10),
                    'capture' => true,
                    'savePaymentMethod' => null,
                    'airline' => [
                        'booking_reference' => Random::str(10),
                        'ticket_number' => Random::int(10),
                        'passengers' => [
                            [
                                'first_name' => Random::str(10, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                                'last_name' => Random::str(10, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                            ],
                        ],
                        'legs' => [
                            [
                                'departure_airport' => Random::str(3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                                'destination_airport' => Random::str(3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                                'departure_date' => '2020-01-01',
                            ],
                        ],
                    ],
                    'transfers' => [
                        [
                            'account_id' => (string) Random::int(11111111, 99999999),
                            'amount' => [
                                'value' => sprintf('%.2f', round($amount, 2)),
                                'currency' => Random::value(CurrencyCode::getValidValues()),
                            ],
                            'status' => Random::value(TransferStatus::getValidValues()),
                            'platform_fee_amount' => [
                                'value' => sprintf('%.2f', round(Random::float(0.1, 99.99), 2)),
                                'currency' => Random::value(CurrencyCode::getValidValues()),
                            ],
                            'description' => Random::str(1, Transfer::MAX_LENGTH_DESCRIPTION),
                            'metadata' => ['test' => Random::str(10, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')],
                            'release_funds' => Random::bool(),
                        ],
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
                    'merchant_customer_id' => Random::str(36, Payment::MAX_LENGTH_MERCHANT_CUSTOMER_ID),
                ],
            ],
        ];
        $confirmations = [
            new ConfirmationAttributesExternal(),
            new ConfirmationAttributesRedirect(),
            new ConfirmationAttributesMobileApplication(),
        ];
        $paymentData = [
            new PaymentDataSbp(),
            new PaymentDataApplePay(),
            new PaymentDataGooglePay(),
            new PaymentDataBankCard(),
            new PaymentDataMobileBalance(['phone' => Random::str(11, '0123456789')]),
            new PaymentDataQiwi(['phone' => Random::str(11, '0123456789')]),
            new PaymentDataSberbank(['phone' => Random::str(11, '0123456789')]),
            new PaymentDataCash(['phone' => Random::str(11, '0123456789')]),
            new PaymentDataYooMoney(),
            new PaymentDataInstallments(),
            new PaymentDataB2bSberbank(['payment_purpose' => Random::str(16), 'vat_data' => ['type' => VatDataType::UNTAXED]]),
        ];

        $paymentData[1]->setPaymentData(Random::str(10));
        $paymentData[2]->setPaymentMethodToken(Random::str(10));
        $paymentData[2]->setGoogleTransactionId(Random::str(10));

        $card = new PaymentDataBankCardCard();
        $card->setNumber(Random::str(16, '0123456789'));
        $card->setExpiryYear(Random::int(2000, 2200));
        $card->setExpiryMonth(Random::value(['01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12']));
        $card->setCsc(Random::str(4, '0123456789'));
        $card->setCardholder(Random::str(26, 'abcdefghijklmnopqrstuvwxyz'));
        $paymentData[3]->setCard($card);
        $paymentData[4]->setPhone(Random::str(14, '0123456789'));

        $paymentData[6]->setPhone(Random::str(14, '0123456789'));

        /** @var PaymentDataB2bSberbank $paymentData [10] */
        $paymentDataB2bSberbank = new PaymentDataB2bSberbank();
        $paymentDataB2bSberbank->setPaymentPurpose(Random::str(10));
        $paymentDataB2bSberbank->setVatData([
            'type' => VatDataType::CALCULATED,
            'rate' => VatDataRate::RATE_10,
            'amount' => [
                'value' => Random::int(1, 10000),
                'currency' => CurrencyCode::USD,
            ],
        ]);
        $paymentData[10] = $paymentDataB2bSberbank;

        $confirmations[0]->setLocale('en_US');
        $confirmations[1]->setEnforce(true);
        $confirmations[1]->setReturnUrl('https://test.com');
        $confirmations[2]->setReturnUrl('https://test.ru');
        foreach ($paymentData as $i => $paymentMethodData) {
            $request = [
                'accountId' => uniqid('', true),
                'gatewayId' => uniqid('', true),
                'amount' => [
                    'value' => sprintf('%.2f', round($amount, 2)),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
                'referenceId' => uniqid('', true),
                'paymentMethodData' => $paymentData[$i],
                'confirmation' => Random::value($confirmations),
                'savePaymentMethod' => Random::bool(),
                'capture' => (bool)Random::int(0, 1),
                'clientIp' => long2ip(Random::int(0, 2 ** 32)),
                'metadata' => ['test' => uniqid('', true)],
                'receipt' => [
                    'items' => $this->getReceiptItem($i + 1),
                    'customer' => [
                        'email' => 'johndoe@yoomoney.ru',
                        'phone' => Random::str(12, '0123456789'),
                    ],
                    'tax_system_code' => Random::int(1, 6),
                ],
                'airline' => $airline->toArray(),
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
                'merchant_customer_id' => Random::str(36, Payment::MAX_LENGTH_MERCHANT_CUSTOMER_ID),
            ];
            $result[] = [$request];
        }

        return $result;
    }

    /**
     * @param mixed $count
     *
     * @return array
     */
    private function getReceiptItem(mixed $count): array
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
                'payment_subject' => PaymentSubject::COMMODITY,
                'payment_mode' => PaymentMode::PARTIAL_PREPAYMENT,
                'payment_subject_industry_details' => [
                    [
                        'federal_id' => '001',
                        'document_date' => date('Y-m-d', Random::int(100000000, 200000000)),
                        'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                        'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                    ]
                ]
            ];
        }

        return $result;
    }
}
