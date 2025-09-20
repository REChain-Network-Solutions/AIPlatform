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

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use TypeError;
use YooKassa\Common\ListObject;
use YooKassa\Helpers\ProductCode;
use YooKassa\Helpers\Random;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\ReceiptItemMeasure;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\SettlementType;
use YooKassa\Request\Payments\Airline;
use YooKassa\Request\Receipts\ReceiptResponseInterface;
use YooKassa\Request\Receipts\ReceiptResponseItem;
use YooKassa\Request\Receipts\ReceiptResponseItemInterface;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * AbstractTestReceiptResponse
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestReceiptResponse extends TestCase
{
    protected string $type;

    protected bool $valid = true;

    /**
     * @dataProvider validDataProvider
     */
    abstract public function testSpecificProperties(array $options);

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
        if (empty($options['status'])) {
            self::assertNull($instance->getStatus());
        } else {
            self::assertEquals($options['status'], $instance->getStatus());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetTaxSystemCode(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['tax_system_code'])) {
            self::assertNull($instance->getTaxSystemCode());
        } else {
            self::assertEquals($options['tax_system_code'], $instance->getTaxSystemCode());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetReceiptOperationalDetails(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['receipt_operational_details'])) {
            self::assertNull($instance->getReceiptOperationalDetails());
        } else {
            self::assertEquals($options['receipt_operational_details'], $instance->getReceiptOperationalDetails()->toArray());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testReceiptIndustryDetails(array $options): void
    {
        $instance = $this->getTestInstance($options);

        self::assertCount(count($options['receipt_industry_details']), $instance->getReceiptIndustryDetails());

        foreach ($instance->getReceiptIndustryDetails() as $index => $item) {
            self::assertInstanceOf(IndustryDetails::class, $item);
            self::assertArrayHasKey($index, $options['receipt_industry_details']);
            self::assertEquals($options['receipt_industry_details'][$index]['federal_id'], $item->getFederalId());
            self::assertEquals($options['receipt_industry_details'][$index]['document_date'], $item->getDocumentDate()->format(IndustryDetails::DOCUMENT_DATE_FORMAT));
            self::assertEquals($options['receipt_industry_details'][$index]['document_number'], $item->getDocumentNumber());
            self::assertEquals($options['receipt_industry_details'][$index]['value'], $item->getValue());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetItems(array $options): void
    {
        $instance = $this->getTestInstance($options);

        self::assertEquals(count($options['items']), count($instance->getItems()));

        foreach ($instance->getItems() as $index => $item) {
            self::assertInstanceOf(ReceiptResponseItemInterface::class, $item);
            self::assertArrayHasKey($index, $options['items']);
            self::assertEquals($options['items'][$index]['description'], $item->getDescription());
            self::assertEquals($options['items'][$index]['amount']['value'], $item->getPrice()->getValue());
            self::assertEquals($options['items'][$index]['amount']['currency'], $item->getPrice()->getCurrency());
            self::assertEquals($options['items'][$index]['quantity'], $item->getQuantity());
            self::assertEquals($options['items'][$index]['vat_code'], $item->getVatCode());
        }
    }

    public function validDataProvider(): array
    {
        date_default_timezone_set('UTC');
        $this->valid = true;
        $receipts = [];
        for ($i = 0; $i < 10; $i++) {
            $receipts[] = $this->generateReceipts($this->type, true);
        }

        return $receipts;
    }

    public function invalidDataProvider(): array
    {
        $this->valid = false;
        $receipts = [];
        for ($i = 0; $i < 10; $i++) {
            $receipts[] = $this->generateReceipts($this->type, false);
        }

        return $receipts;
    }

    public function invalidAllDataProvider(): array
    {
        return [
            [[new ProductCode()]],
            [[new Airline()]],
            [SettlementType::PREPAYMENT],
            [0],
            ['test'],
            [10],
        ];
    }

    public function invalidBoolDataProvider(): array
    {
        return [
            [true],
            [false],
        ];
    }

    public function invalidBoolNullDataProvider(): array
    {
        return [
            [new \stdClass()],
        ];
    }

    public function invalidSettlementsDataProvider(): array
    {
        return [
            [[[]], EmptyPropertyValueException::class],
            [Random::str(10), TypeError::class],
        ];
    }

    public function invalidItemsDataProvider(): array
    {
        return [
            [[[]], EmptyPropertyValueException::class],
            [Random::str(10), TypeError::class],
        ];
    }

    public function invalidFromArray(): array
    {
        return [
            [
                [
                    'id' => Random::str(39),
                    'type' => Random::value(ReceiptType::getEnabledValues()),
                    'status' => null,
                    'items' => false,
                ],
            ],
            [
                [
                    'id' => Random::str(39),
                    'type' => Random::value(ReceiptType::getEnabledValues()),
                    'status' => null,
                    'items' => 1,
                ],
            ],
            [
                [
                    'id' => Random::str(39),
                    'type' => Random::value(ReceiptType::getValidValues()),
                    'status' => null,
                    'items' => [new ReceiptResponseItem()],
                    'settlements' => 1,
                ],
            ],
        ];
    }

    /**
     * @param mixed $options
     */
    abstract protected function getTestInstance(mixed $options): ReceiptResponseInterface;

    /**
     * @param mixed $i
     * @param mixed $options
     */
    abstract protected function addSpecificProperties(mixed $options, mixed $i): array;

    private function generateReceipts($type, $valid): array
    {
        $this->valid = $valid;
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateReceipt($type, $i);
        }

        return $return;
    }

    private function generateReceipt($type, $index): array
    {
        $receipt = [
            'id' => Random::str(39),
            'type' => $type,
            'status' => Random::value(['pending', 'succeeded', 'canceled']),
            'fiscal_document_number' => Random::int(4),
            'fiscal_storage_number' => Random::int(16),
            'fiscal_attribute' => Random::int(10),
            'registered_at' => date(YOOKASSA_DATE, Random::int(1000000000, time())),
            'fiscal_provider_id' => Random::str(36),
            'items' => $this->generateItems(),
            'settlements' => $this->generateSettlements(),
            'tax_system_code' => Random::int(1, 6),
            'receipt_industry_details' => [
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
            'receipt_operational_details' => [
                'operation_id' => Random::int(1, OperationalDetails::OPERATION_ID_MAX_VALUE),
                'value' => Random::str(1, OperationalDetails::VALUE_MAX_LENGTH),
                'created_at' => date(YOOKASSA_DATE, Random::int(1000000000, time())),
            ],
            'on_behalf_of' => Random::int(6),
        ];

        return $this->addSpecificProperties($receipt, $index);
    }

    private function generateItems(): array
    {
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateItem();
        }

        return $return;
    }

    private function generateItem(): array
    {
        $item = [
            'description' => Random::str(1, 128),
            'amount' => [
                'value' => round(Random::float(1.00, 100.00), 2),
                'currency' => 'RUB',
            ],
            'quantity' => round(Random::float(0.001, 99.999), 3),
            'measure' => Random::value(ReceiptItemMeasure::getValidValues()),
            'vat_code' => Random::int(1, 6),
            'country_of_origin_code' => Random::value(['RU', 'US', 'CN']),
            'customs_declaration_number' => Random::str(1, 32),
            'mark_code_info' => [
                'mark_code_raw' => '010460406000590021N4N57RTCBUZTQ\u001d2403054002410161218\u001d1424010191ffd0\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==',
            ],
            'mark_mode' => 0,
            'payment_subject_industry_details' => [
                [
                    'federal_id' => '001',
                    'document_date' => date('Y-m-d', Random::int(100000000, 200000000)),
                    'document_number' => Random::str(1, IndustryDetails::DOCUMENT_NUMBER_MAX_LENGTH),
                    'value' => Random::str(1, IndustryDetails::VALUE_MAX_LENGTH),
                ],
            ],
        ];
        if (ReceiptItemMeasure::PIECE === $item['measure']) {
            $item['mark_quantity'] = [
                'numerator' => Random::int(1, 100),
                'denominator' => 100,
            ];
        }

        return $item;
    }

    private function generateSettlements(): array
    {
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateSettlement();
        }

        return $return;
    }

    private function generateSettlement(): array
    {
        return [
            'type' => Random::value(SettlementType::getValidValues()),
            'description' => Random::str(1, 128),
            'amount' => [
                'value' => round(Random::float(1.00, 100.00), 2),
                'currency' => 'RUB',
            ],
            'quantity' => round(Random::float(0.001, 99.999), 3),
            'vat_code' => Random::int(1, 6),
        ];
    }
}
