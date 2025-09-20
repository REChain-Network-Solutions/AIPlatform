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
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use stdClass;
use YooKassa\Helpers\Random;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\SettlementPayoutPaymentType;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Model\Payment\Payment;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesExternal;
use YooKassa\Request\Payments\CreatePaymentRequestBuilder;
use YooKassa\Request\Payments\PaymentData\PaymentDataQiwi;
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderType;
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities;
use YooKassa\Request\Payments\ReceiverData\ReceiverDigitalWallet;
use YooKassa\Request\Payments\ReceiverData\ReceiverType;
use YooKassa\Request\Payments\Recipient;
use YooKassa\Validator\Exceptions\ValidatorParameterException;

/**
 * CreatePaymentRequestBuilderTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePaymentRequestBuilderTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetDeal(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();
        $builder->setAmount($options['amount']);
        $builder->setDeal($options['deal']);
        $instance = $builder->build();

        if (empty($options['deal'])) {
            self::assertNull($instance->getDeal());
        } else {
            self::assertNotNull($instance->getDeal());
            self::assertEquals($options['deal'], $instance->getDeal()->toArray());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetMerchantCustomerId(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();
        $builder->setAmount($options['amount']);
        $builder->setMerchantCustomerId($options['merchant_customer_id']);
        $instance = $builder->build();

        if (empty($options['merchant_customer_id'])) {
            self::assertNull($instance->getMerchantCustomerId());
        } else {
            self::assertNotNull($instance->getMerchantCustomerId());
            self::assertEquals($options['merchant_customer_id'], $instance->getMerchantCustomerId());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetProductGroupId(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getRecipient());

        $builder->setGatewayId($options['gatewayId']);
        $instance = $builder->build($this->getRequiredData('gatewayId'));

        if (empty($options['gatewayId'])) {
            self::assertNull($instance->getRecipient());
        } else {
            self::assertNotNull($instance->getRecipient());
            self::assertEquals($options['gatewayId'], $instance->getRecipient()->getGatewayId());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetAmount(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNotNull($instance->getAmount());

        $builder->setAmount($options['amount']);
        $instance = $builder->build($this->getRequiredData('amount'));

        if ($options['amount'] instanceof AmountInterface) {
            self::assertEquals($options['amount']->getValue(), $instance->getAmount()->getValue());
            self::assertEquals($options['amount']->getCurrency(), $instance->getAmount()->getCurrency());
        } else {
            self::assertEquals($options['amount']['value'], $instance->getAmount()->getValue());
            self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
        }

        if (is_array($options['amount'])) {
            $builder->setAmount($options['amount']['value'], $options['amount']['currency']);
            $instance = $builder->build($this->getRequiredData('amount'));
            self::assertEquals($options['amount']['value'], $instance->getAmount()->getValue());
            self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
        }

        $builder->setAmount(10000)->setAmount($options['amount']);
        $instance = $builder->build($this->getRequiredData('amount'));

        if ($options['amount'] instanceof AmountInterface) {
            self::assertEquals($options['amount']->getValue(), $instance->getAmount()->getValue());
            self::assertEquals($options['amount']->getCurrency(), $instance->getAmount()->getCurrency());
        } else {
            self::assertEquals($options['amount']['value'], $instance->getAmount()->getValue());
            self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
        }

        if (!$options['amount'] instanceof AmountInterface) {
            $builder->setAmount([
                'value' => $options['amount']['value'],
                'currency' => $options['amount']['currency'],
            ]);
            $instance = $builder->build($this->getRequiredData('amount'));
            self::assertEquals($options['amount']['value'], $instance->getAmount()->getValue());
            self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
        }
    }

    /**
     * @dataProvider invalidAmountDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidAmount(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setAmount($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCurrency(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNotNull($instance->getAmount());
        self::assertEquals(CurrencyCode::RUB, $instance->getAmount()->getCurrency());

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setAmount($options['amount']);
        $builder->setCurrency($options['amount']['currency']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $instance = $builder->build($this->getRequiredData('amount'));

        if ($options['amount'] instanceof AmountInterface) {
            self::assertEquals($options['amount']->getCurrency(), $instance->getAmount()->getCurrency());
        } else {
            self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
        }
        if (!empty($options['receiptItems'])) {
            foreach ($instance->getReceipt()->getItems() as $item) {
                self::assertEquals($options['amount']['currency'], $item->getPrice()->getCurrency());
            }
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptItems(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals(count($options['receiptItems']), count($instance->getReceipt()->getItems()));
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testAddReceiptItems(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        foreach ($options['receiptItems'] as $item) {
            if ($item instanceof ReceiptItem) {
                $builder->addReceiptItem(
                    $item->getDescription(),
                    $item->getPrice()->getValue(),
                    $item->getQuantity(),
                    $item->getVatCode(),
                    $item->getPaymentMode(),
                    $item->getPaymentSubject()
                );
            } else {
                $builder->addReceiptItem(
                    $item['description'],
                    $item['price']['value'],
                    $item['quantity'],
                    $item['vatCode'],
                    $item['paymentMode'],
                    $item['paymentSubject']
                );
            }
        }
        $builder->setReceiptEmail($options['receiptEmail']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals(count($options['receiptItems']), count($instance->getReceipt()->getItems()));
            foreach ($instance->getReceipt()->getItems() as $item) {
                self::assertFalse($item->isShipping());
            }
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testAddReceiptShipping(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        foreach ($options['receiptItems'] as $item) {
            if ($item instanceof ReceiptItem) {
                $builder->addReceiptShipping(
                    $item->getDescription(),
                    $item->getPrice()->getValue(),
                    $item->getVatCode()
                );
            } else {
                $builder->addReceiptShipping($item['description'], $item['price']['value'], $item['vatCode']);
            }
        }
        $builder->setReceiptEmail($options['receiptEmail']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals(count($options['receiptItems']), count($instance->getReceipt()->getItems()));
            foreach ($instance->getReceipt()->getItems() as $item) {
                self::assertTrue($item->isShipping());
            }
        }
    }

    /**
     * @dataProvider invalidItemsDataProvider
     *
     * @param mixed $items
     */
    public function testSetInvalidReceiptItems(mixed $items): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceiptItems($items);
    }

    public static function invalidItemsDataProvider(): array
    {
        return [
            [
                [
                    [
                        'price' => [1],
                        'quantity' => -1.4,
                        'vatCode' => 11,
                    ],
                ],
            ],
            [
                [
                    [
                        'title' => 'test',
                        'quantity' => -1.4,
                        'vatCode' => 3,
                    ],
                ],
            ],
            [
                [
                    [
                        'description' => 'test',
                        'quantity' => 1.4,
                        'vatCode' => -3,
                    ],
                ],
            ],
            [
                [
                    [
                        'title' => 'test',
                        'price' => [123],
                        'quantity' => 1.4,
                        'vatCode' => 12,
                    ],
                ],
            ],
            [
                [
                    [
                        'description' => 'test',
                        'price' => [123],
                        'quantity' => -1.4,
                    ],
                ],
            ],
            [
                [
                    [
                        'title' => 'test',
                        'price' => [1],
                        'vatCode' => 20,
                    ],
                ],
            ],
        ];
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptEmail(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals($options['receiptEmail'], $instance->getReceipt()->getCustomer()->getEmail());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptPhone(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $builder->setReceiptPhone($options['receiptPhone']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals($options['receiptPhone'], $instance->getReceipt()->getCustomer()->getPhone());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptTaxSystemCode(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $builder->setTaxSystemCode($options['taxSystemCode']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals($options['taxSystemCode'], $instance->getReceipt()->getTaxSystemCode());
        }
    }

    /**
     * @dataProvider invalidVatIdDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidTaxSystemId(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setTaxSystemCode($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptIndustryDetails(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $builder->setReceiptIndustryDetails($options['receiptIndustryDetails']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals($options['receiptIndustryDetails'], $instance->getReceipt()->getReceiptIndustryDetails()->toArray());
        }
    }

    /**
     * @dataProvider invalidReceiptIndustryDetailsDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidReceiptIndustryDetails(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceiptIndustryDetails($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiptOperationalDetails(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $builder->setReceiptOperationalDetails($options['receiptOperationalDetails']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiptItems'])) {
            self::assertNull($instance->getReceipt());
        } else {
            self::assertNotNull($instance->getReceipt());
            self::assertEquals($options['receiptOperationalDetails'], $instance->getReceipt()->getReceiptOperationalDetails());
        }
    }

    /**
     * @dataProvider invalidReceiptOperationalDetailsDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidReceiptOperationalDetails(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceiptOperationalDetails($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetPaymentToken(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData(null, 'paymentMethodId'));
        self::assertNull($instance->getPaymentToken());
        self::assertNull($instance->paymentToken);
        self::assertNull($instance->payment_token);

        if (empty($options['paymentToken'])) {
            $buildData = $this->getRequiredData(null, 'paymentMethodId');
        } else {
            $buildData = $this->getRequiredData('paymentToken');
        }

        $builder->setPaymentToken($options['paymentToken']);
        $instance = $builder->build($buildData);

        if (empty($options['paymentToken'])) {
            self::assertNull($instance->getPaymentToken());
            self::assertNull($instance->paymentToken);
            self::assertNull($instance->payment_token);
        } else {
            self::assertEquals($options['paymentToken'], $instance->getPaymentToken());
            self::assertEquals($options['paymentToken'], $instance->paymentToken);
            self::assertEquals($options['paymentToken'], $instance->payment_token);
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetPaymentMethodId(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getPaymentMethodId());

        $builder->setPaymentMethodId($options['paymentMethodId']);
        $instance = $builder->build($this->getRequiredData(empty($options['paymentMethodId']) ? null : 'paymentToken'));

        if (empty($options['paymentMethodId'])) {
            self::assertNull($instance->getPaymentMethodId());
        } else {
            self::assertEquals($options['paymentMethodId'], $instance->getPaymentMethodId());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetPaymentData(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getPaymentMethodData());

        $builder->setPaymentMethodData($options['paymentMethodData']);
        $instance = $builder->build($this->getRequiredData(empty($options['paymentMethodId']) ? null : 'paymentToken'));

        if (empty($options['paymentMethodData'])) {
            self::assertNull($instance->getPaymentMethodData());
        } else {
            if (is_object($options['paymentMethodData'])) {
                self::assertSame($options['paymentMethodData'], $instance->getPaymentMethodData());
            } elseif (is_string($options['paymentMethodData'])) {
                self::assertEquals($options['paymentMethodData'], $instance->getPaymentMethodData()->getType());
            } else {
                self::assertEquals($options['paymentMethodData']['type'], $instance->getPaymentMethodData()->getType());
            }
        }

        if (is_array($options['paymentMethodData'])) {
            $builder = new CreatePaymentRequestBuilder();
            $builder->build($this->getRequiredData());
            $builder->setPaymentMethodData($options['paymentMethodData']['type'], $options['paymentMethodData']);
            $instance = $builder->build($this->getRequiredData(empty($options['paymentMethodId']) ? null : 'paymentToken'));
            self::assertEquals($options['paymentMethodData']['type'], $instance->getPaymentMethodData()->getType());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetConfirmationAttributes(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getConfirmation());

        $builder->setConfirmation($options['confirmation']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['confirmation'])) {
            self::assertNull($instance->getConfirmation());
        } else {
            if (is_object($options['confirmation'])) {
                self::assertSame($options['confirmation'], $instance->getConfirmation());
            } elseif (is_string($options['confirmation'])) {
                self::assertEquals($options['confirmation'], $instance->getConfirmation()->getType());
            } else {
                self::assertEquals($options['confirmation']['type'], $instance->getConfirmation()->getType());
                self::assertEquals($options['confirmation']['locale'], $instance->getConfirmation()->getLocale());
            }
        }

        if (is_array($options['confirmation'])) {
            $builder = new CreatePaymentRequestBuilder();
            $builder->build($this->getRequiredData());
            $builder->setConfirmation($options['confirmation']);
            $instance = $builder->build($this->getRequiredData());
            self::assertEquals($options['confirmation']['type'], $instance->getConfirmation()->getType());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCreateRecurring(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getSavePaymentMethod());

        $builder->setSavePaymentMethod($options['savePaymentMethod']);
        $instance = $builder->build($this->getRequiredData());

        if (null === $options['savePaymentMethod'] || '' === $options['savePaymentMethod']) {
            self::assertFalse($instance->getSavePaymentMethod());
        } else {
            self::assertEquals($options['savePaymentMethod'], $instance->getSavePaymentMethod());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetCapture(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertTrue($instance->getCapture());

        $builder->setCapture($options['capture']);
        $instance = $builder->build($this->getRequiredData());

        if (null === $options['capture'] || '' === $options['capture']) {
            self::assertTrue($instance->getCapture());
        } else {
            self::assertEquals($options['capture'], $instance->getCapture());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetClientIp(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getClientIp());

        $builder->setClientIp($options['clientIp']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['clientIp'])) {
            self::assertNull($instance->getClientIp());
        } else {
            self::assertEquals($options['clientIp'], $instance->getClientIp());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetMetadata(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $instance = $builder->build($this->getRequiredData());
        self::assertNull($instance->getMetadata());

        $builder->setMetadata($options['metadata']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['metadata'])) {
            self::assertNull($instance->getMetadata());
        } else {
            self::assertEquals($options['metadata'], $instance->getMetadata()->toArray());
        }
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetRecipient(mixed $options): void
    {
        $recipient = new Recipient();
        $recipient->setGatewayId($options['gatewayId']);

        $builder = new CreatePaymentRequestBuilder();
        $builder->setRecipient($recipient);
        $instance = $builder->build($this->getRequiredData());

        self::assertEquals($recipient, $instance->getRecipient());

        $builder = new CreatePaymentRequestBuilder();
        $builder->setRecipient([
            'gateway_id' => $options['gatewayId'],
        ]);
        $instance = $builder->build($this->getRequiredData());

        self::assertEquals($recipient, $instance->getRecipient());
    }

    /**
     * @dataProvider invalidRecipientDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidRecipient(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setRecipient($value);
    }

    public static function invalidRecipientDataProvider(): array
    {
        return [
            [null],
            [true],
            [false],
            [1],
            [1.1],
            ['test'],
            [new stdClass()],
        ];
    }

    /**
     * @throws Exception
     */
    public function testSetReceipt(): void
    {
        $receipt = [
            'tax_system_code' => Random::int(1, 6),
            'customer' => [
                'email' => 'johndoe@yoomoney.ru',
                'phone' => Random::str(4, 15, '0123456789'),
            ],
            'items' => [
                [
                    'description' => 'test',
                    'quantity' => 123,
                    'amount' => [
                        'value' => 321,
                        'currency' => 'USD',
                    ],
                    'vat_code' => Random::int(1, 6),
                    'payment_subject' => PaymentSubject::COMMODITY,
                    'payment_mode' => PaymentMode::PARTIAL_PREPAYMENT,
                ],
            ],
        ];

        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceipt($receipt);
        $instance = $builder->build($this->getRequiredData());

        self::assertEquals($receipt['tax_system_code'], $instance->getReceipt()->getTaxSystemCode());
        self::assertEquals($receipt['customer']['email'], $instance->getReceipt()->getCustomer()->getEmail());
        self::assertEquals($receipt['customer']['phone'], $instance->getReceipt()->getCustomer()->getPhone());
        self::assertCount(1, $instance->getReceipt()->getItems());

        $receipt = $instance->getReceipt();

        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceipt($instance->getReceipt());
        $instance = $builder->build($this->getRequiredData());

        self::assertEquals($receipt['tax_system_code'], $instance->getReceipt()->getTaxSystemCode());
        self::assertEquals($receipt['customer']['email'], $instance->getReceipt()->getCustomer()->getEmail());
        self::assertEquals($receipt['customer']['phone'], $instance->getReceipt()->getCustomer()->getPhone());
        self::assertCount(1, $instance->getReceipt()->getItems());
    }

    /**
     * @dataProvider invalidReceiptDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidReceipt(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceipt($value);
    }

    public static function invalidReceiptDataProvider(): array
    {
        return [
            [null],
            [true],
            [false],
            [1],
            [1.1],
            ['test'],
            [new stdClass()],
        ];
    }

    /**
     * @throws Exception
     */
    public static function validDataProvider(): array
    {
        $receiptItem = new ReceiptItem();
        $receiptItem->setPrice(new ReceiptItemAmount(1));
        $receiptItem->setQuantity(1);
        $receiptItem->setDescription('test');
        $receiptItem->setVatCode(3);
        $result = [
            [
                [
                    'accountId' => Random::str(1, 32),
                    'gatewayId' => Random::str(1, 32),
                    'recipient' => null,
                    'description' => Random::str(1, 128),
                    'amount' => new MonetaryAmount(Random::int(1, 1000)),
                    'receiptItems' => [],
                    'paymentToken' => null,
                    'paymentMethodId' => null,
                    'paymentMethodData' => null,
                    'confirmation' => null,
                    'savePaymentMethod' => true,
                    'capture' => true,
                    'clientIp' => null,
                    'metadata' => null,
                    'receiptEmail' => 'johndoe@yoomoney.ru',
                    'receiptPhone' => null,
                    'taxSystemCode' => null,
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
                    'merchant_customer_id' => null,
                    'receiptIndustryDetails' => [],
                    'receiptOperationalDetails' => null,
                    'receiver' => null,
                    'paymentOrder' => null,
                ],
            ],
            [
                [
                    'accountId' => Random::str(1, 32),
                    'gatewayId' => Random::str(1, 32),
                    'recipient' => null,
                    'description' => Random::str(1, 128),
                    'amount' => new MonetaryAmount(Random::int(1, 1000)),
                    'receiptItems' => [
                        [
                            'description' => 'test',
                            'quantity' => Random::int(1, 100),
                            'price' => (new MonetaryAmount(Random::int(1, 1000)))->toArray(),
                            'vatCode' => Random::int(1, 6),
                            'paymentMode' => PaymentMode::CREDIT_PAYMENT,
                            'paymentSubject' => PaymentSubject::ANOTHER,
                        ],
                        $receiptItem,
                    ],
                    'referenceId' => null,
                    'paymentToken' => null,
                    'paymentMethodId' => null,
                    'paymentMethodData' => null,
                    'confirmation' => null,
                    'savePaymentMethod' => false,
                    'capture' => false,
                    'clientIp' => '',
                    'metadata' => [],
                    'receiptEmail' => 'johndoe@yoomoney.ru',
                    'receiptPhone' => '',
                    'taxSystemCode' => null,
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
                    'merchant_customer_id' => null,
                    'receiptIndustryDetails' => [],
                    'receiptOperationalDetails' => null,
                    'receiver' => null,
                    'paymentOrder' => null,
                ],
            ],
        ];
        $paymentMethodData = [
            new PaymentDataQiwi(['phone' => Random::str(11, '0123456789')]),
            PaymentMethodType::BANK_CARD,
            [
                'type' => PaymentMethodType::BANK_CARD,
            ],
        ];
        $confirmationStatuses = [
            new ConfirmationAttributesExternal(),
            [
                'type' => ConfirmationType::EMBEDDED,
                'locale' => 'en_US',
            ],
            [
                'type' => ConfirmationType::EXTERNAL,
                'locale' => 'en_US',
            ],
            [
                'type' => ConfirmationType::QR,
                'locale' => 'ru_RU',
            ],
        ];
        $receivers = [
            [
                'type' => ReceiverType::BANK_ACCOUNT,
                'account_number' => Random::str(20, 20, '0123456789'),
                'bic' => Random::str(9, 9, '0123456789'),
            ],
            new ReceiverDigitalWallet(['account_number' => Random::str(20, 20, '0123456789')]),
            [
                'type' => ReceiverType::MOBILE_BALANCE,
                'phone' => Random::str(4, 15, '0123456789'),
            ],
        ];
        $paymentOrderArr = [
            'type' => PaymentOrderType::UTILITIES,
            'amount' => [
                'value' => Random::int(1, 100000),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'payment_purpose' => Random::str(1, 210),
            'recipient' => [
                'name' => Random::str(1, 100),
                'inn' => Random::str(10, 10, '0123456789'),
                'kpp' => Random::str(9, 9, '0123456789'),
                'bank' => [
                    'name' => Random::str(1, 100),
                    'bic' => Random::str(9, 9, '0123456789'),
                    'account' => Random::str(20, 20, '0123456789'),
                    'correspondent_account' => Random::str(20, 20, '0123456789'),
                ]
            ],
            'kbk' => Random::str(20, 20, '0123456789'),
            'oktmo' => Random::str(8, 8, '0123456789'),
            'payment_period' => [
                'month' => Random::int(1, 12),
                'year' => Random::int(1900, 3000),
            ],
            'payment_document_id' => Random::str(18),
            'payment_document_number' => '34СТ185329012352',
            'account_number' => '34СТ185329012352',
            'unified_account_number' => Random::str(10),
            'service_id' => Random::str(13),
        ];
        $paymentOrders = [$paymentOrderArr, new PaymentOrderUtilities($paymentOrderArr)];
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'accountId' => uniqid('', true),
                'gatewayId' => uniqid('', true),
                'recipient' => new Recipient(),
                'description' => uniqid('', true),
                'amount' => [
                    'value' => Random::int(1, 100000),
                    'currency' => Random::value(CurrencyCode::getValidValues()),
                ],
                'receiptItems' => [],
                'referenceId' => uniqid('', true),
                'paymentToken' => uniqid('', true),
                'paymentMethodId' => uniqid('', true),
                'paymentMethodData' => $paymentMethodData[$i] ?? null,
                'confirmation' => $confirmationStatuses[$i] ?? null,
                'savePaymentMethod' => (bool)Random::int(0, 1),
                'capture' => (bool)Random::int(0, 1),
                'clientIp' => long2ip(Random::int(0, 2 ** 32)),
                'metadata' => ['test' => 'test'],
                'receiptEmail' => 'johndoe@yoomoney.ru',
                'receiptPhone' => Random::str(10, '0123456789'),
                'taxSystemCode' => Random::int(1, 6),
                'receiptIndustryDetails' => [
                    [
                        'federal_id' => Random::value([
                            '00' . Random::int(1, 9),
                            '0' . Random::int(1, 6) . Random::int(0, 9),
                            '07' . Random::int(0, 3)
                        ]),
                        'document_date' => date(IndustryDetails::DOCUMENT_DATE_FORMAT),
                        'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                        'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                    ],
                ],
                'receiptOperationalDetails' => [
                    'operation_id' => Random::int(0, OperationalDetails::OPERATION_ID_MAX_VALUE),
                    'value' => Random::str(1, OperationalDetails::VALUE_MAX_LENGTH),
                    'created_at' => date(YOOKASSA_DATE),
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
                'merchant_customer_id' => Random::str(3, 100),
                'receiver' => Random::value($receivers),
                'paymentOrder' => Random::value($paymentOrders),
            ];
            $result[] = [$request];
        }

        return $result;
    }

    public static function invalidAmountDataProvider(): array
    {
        return [
            [-1],
            [true],
            [false],
            [new stdClass()],
            [0],
        ];
    }

    /**
     * @throws Exception
     */
    public static function invalidVatIdDataProvider(): array
    {
        return [
            [false],
            [0],
            [7],
            [Random::int(-100, -1)],
            [Random::int(7, 100)],
        ];
    }

    /**
     * @throws Exception
     */
    public static function invalidReceiptIndustryDetailsDataProvider(): array
    {
        return [
            [new stdClass()],
            [true],
            [Random::str(1, 100)],
        ];
    }

    public static function invalidReceiptOperationalDetailsDataProvider(): array
    {
        return [
            [new stdClass()],
            [true],
            [Random::str(1, 100)],
        ];
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetDescription(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setDescription($options['description']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['description'])) {
            self::assertNull($instance->getDescription());
        } else {
            self::assertEquals($options['description'], $instance->getDescription());
        }
    }

    public function testSetInvalidLengthDescription(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $description = Random::str(Payment::MAX_LENGTH_DESCRIPTION + 1);
        $builder->setDescription($description);
    }

    /**
     * @dataProvider invalidVatIdDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidAirline(mixed $value): void
    {
        $this->expectException(ValidatorParameterException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setAirline($value);
    }

    /**
     * @throws Exception
     */
    public static function invalidDealDataProvider(): array
    {
        return [
            [true],
            [false],
            [new stdClass()],
            [0],
            [7],
            [Random::int(-100, -1)],
            [Random::int(7, 100)],
        ];
    }

    /**
     * @dataProvider invalidDealDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidDeal(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setDeal($value);
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetReceiver(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();

        $builder->setReceiptItems($options['receiptItems']);
        $builder->setReceiptEmail($options['receiptEmail']);
        $builder->setReceiver($options['receiver']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['receiver'])) {
            self::assertNull($instance->getReceiver());
        } else {
            self::assertNotNull($instance->getReceiver());
            self::assertEquals($options['receiver'], is_array($options['receiver']) ? $instance->getReceiver()->toArray() : $instance->getReceiver());
        }
    }

    /**
     * @dataProvider invalidReceiverDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidReceiver(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setReceiver($value);
    }

    /**
     * @throws Exception
     */
    public static function invalidReceiverDataProvider(): array
    {
        return [
            [true],
            [false],
            [new stdClass()],
            [0],
            [7],
            [Random::int(-100, -1)],
            [Random::int(7, 100)],
        ];
    }

    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     *
     * @throws Exception
     */
    public function testSetPaymentOrder(mixed $options): void
    {
        $builder = new CreatePaymentRequestBuilder();
        $builder->setPaymentOrder($options['paymentOrder']);
        $instance = $builder->build($this->getRequiredData());

        if (empty($options['paymentOrder'])) {
            self::assertNull($instance->getPaymentOrder());
        } else {
            self::assertNotNull($instance->getPaymentOrder());
            self::assertEquals($options['paymentOrder'], is_array($options['paymentOrder']) ? $instance->getPaymentOrder()->toArray() : $instance->getPaymentOrder());
        }
    }

    /**
     * @dataProvider invalidPaymentOrderDataProvider
     *
     * @param mixed $value
     */
    public function testSetInvalidPaymentOrder(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $builder = new CreatePaymentRequestBuilder();
        $builder->setPaymentOrder($value);
    }

    /**
     * @throws Exception
     */
    public static function invalidPaymentOrderDataProvider(): array
    {
        return [
            [true],
            [false],
            [new stdClass()],
            [0],
            [7],
            [Random::int(-100, -1)],
            [Random::int(7, 100)],
        ];
    }

    /**
     * @param null $testingProperty
     * @param null $paymentType
     *
     * @throws Exception
     */
    protected function getRequiredData($testingProperty = null, $paymentType = null): array
    {
        $result = [];
        if ('accountId' === $testingProperty || 'gatewayId' === $testingProperty) {
            if ('accountId' !== $testingProperty) {
                $result['accountId'] = Random::str(1, 32);
            }
            if ('gatewayId' !== $testingProperty) {
                $result['gatewayId'] = Random::str(1, 32);
            }
        }
        if ('amount' !== $testingProperty) {
            $result['amount'] = new MonetaryAmount(Random::int(1, 100));
        }
        if ('paymentToken' !== $testingProperty) {
            if (null !== $paymentType) {
                $result[$paymentType] = Random::str(36);
            } else {
                $result['paymentToken'] = Random::str(36);
            }
        }

        return $result;
    }
}
