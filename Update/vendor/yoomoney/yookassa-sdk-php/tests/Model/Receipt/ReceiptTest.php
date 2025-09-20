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

namespace Tests\YooKassa\Model\Receipt;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\ListObject;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptCustomer;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Model\Receipt\Settlement;
use YooKassa\Model\Receipt\SettlementType;

/**
 * ReceiptTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptTest extends AbstractTestCase
{
    protected Receipt $object;

    /**
     * @return Receipt
     */
    protected function getTestInstance(): Receipt
    {
        return new Receipt();
    }

    /**
     * @return void
     */
    public function testReceiptClassExists(): void
    {
        $this->object = $this->getMockBuilder(Receipt::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Receipt::class));
        $this->assertInstanceOf(Receipt::class, $this->object);
        self::assertNull($this->object->getObjectId());
    }

    /**
     * Test property "customer"
     * @dataProvider validCustomerDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCustomer(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertNotEmpty($instance->getCustomer());
        self::assertNotEmpty($instance->customer);
        $instance->setCustomer($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getCustomer());
            self::assertNotNull($instance->customer);
            self::assertEquals($value, is_array($value) ? $instance->getCustomer()->toArray() : $instance->getCustomer());
            self::assertEquals($value, is_array($value) ? $instance->customer->toArray() : $instance->customer);
        }
    }

    /**
     * Test invalid property "customer"
     * @dataProvider invalidCustomerDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCustomer(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCustomer($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCustomerDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_customer'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCustomerDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_customer'));
    }

    /**
     * Test property "items"
     * @dataProvider validItemsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testItems(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getItems());
        self::assertIsObject($instance->items);
        self::assertCount(0, $instance->getItems());
        self::assertCount(0, $instance->items);
        $instance->setItems($value);
        self::assertNotNull($instance->getItems());
        self::assertNotNull($instance->items);
        self::assertTrue($instance->notEmpty());
        foreach ($value as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getItems()[$key]->toArray());
                self::assertEquals($element, $instance->items[$key]->toArray());
                self::assertIsArray($instance->getItems()[$key]->toArray());
                self::assertIsArray($instance->items[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getItems()->get($key));
                self::assertIsObject($instance->getItems()->get($key));
                self::assertIsObject($instance->items->get($key));
                self::assertIsObject($instance->getItems());
                self::assertIsObject($instance->items);
            }
        }
        self::assertCount(count($value), $instance->getItems());
        self::assertCount(count($value), $instance->items);

        $instance->removeItems();
        self::assertCount(0, $instance->getItems());
        self::assertCount(0, $instance->items);
    }

    /**
     * Test invalid property "items"
     * @dataProvider invalidItemsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidItems(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setItems($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validItemsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_items'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidItemsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_items'));
    }

    /**
     * Test property "settlements"
     * @dataProvider validSettlementsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSettlements(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSettlements());
        self::assertEmpty($instance->settlements);
        self::assertIsObject($instance->getSettlements());
        self::assertIsObject($instance->settlements);
        self::assertCount(0, $instance->getSettlements());
        self::assertCount(0, $instance->settlements);

        if (!empty($value)) {
            self::assertNotNull($instance->getSettlements());
            self::assertNotNull($instance->settlements);
            foreach ($value as $element) {
                if (is_object($element) && !empty($element)) {
                    $instance->addSettlement($element);
                    self::assertEquals($element, $instance->getSettlements()->get(0));
                    self::assertIsObject($instance->getSettlements()->get(0));
                    self::assertIsObject($instance->settlements->get(0));
                    self::assertIsObject($instance->getSettlements());
                    self::assertIsObject($instance->settlements);
                    $instance->getSettlements()->clear();
                }
            }

            $instance->setSettlements($value);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSettlements()[$key]->toArray());
                    self::assertEquals($element, $instance->settlements[$key]->toArray());
                    self::assertIsArray($instance->getSettlements()[$key]->toArray());
                    self::assertIsArray($instance->settlements[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSettlements()->get($key));
                    self::assertIsObject($instance->getSettlements()->get($key));
                    self::assertIsObject($instance->settlements->get($key));
                    self::assertIsObject($instance->getSettlements());
                    self::assertIsObject($instance->settlements);
                }
            }
            self::assertCount(count($value), $instance->getSettlements());
            self::assertCount(count($value), $instance->settlements);
            self::assertGreaterThanOrEqual(1, $instance->getSettlements()->count());
            self::assertGreaterThanOrEqual(1, $instance->settlements->count());
        }
    }

    /**
     * Test invalid property "settlements"
     * @dataProvider invalidSettlementsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSettlements(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSettlements($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_settlements'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_settlements'));
    }

    /**
     * Test property "tax_system_code"
     * @dataProvider validTaxSystemCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTaxSystemCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getTaxSystemCode());
        self::assertEmpty($instance->tax_system_code);
        $instance->setTaxSystemCode($value);
        self::assertEquals($value, is_array($value) ? $instance->getTaxSystemCode()->toArray() : $instance->getTaxSystemCode());
        self::assertEquals($value, is_array($value) ? $instance->tax_system_code->toArray() : $instance->tax_system_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getTaxSystemCode());
            self::assertNotNull($instance->tax_system_code);
            self::assertLessThanOrEqual(6, is_string($instance->getTaxSystemCode()) ? mb_strlen($instance->getTaxSystemCode()) : $instance->getTaxSystemCode());
            self::assertLessThanOrEqual(6, is_string($instance->tax_system_code) ? mb_strlen($instance->tax_system_code) : $instance->tax_system_code);
            self::assertGreaterThanOrEqual(1, is_string($instance->getTaxSystemCode()) ? mb_strlen($instance->getTaxSystemCode()) : $instance->getTaxSystemCode());
            self::assertGreaterThanOrEqual(1, is_string($instance->tax_system_code) ? mb_strlen($instance->tax_system_code) : $instance->tax_system_code);
            self::assertIsNumeric($instance->getTaxSystemCode());
            self::assertIsNumeric($instance->tax_system_code);
        }
    }

    /**
     * Test invalid property "tax_system_code"
     * @dataProvider invalidTaxSystemCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTaxSystemCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTaxSystemCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTaxSystemCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_tax_system_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTaxSystemCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_tax_system_code'));
    }

    /**
     * Test property "additional_user_props"
     * @dataProvider validAdditionalUserPropsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAdditionalUserProps(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAdditionalUserProps());
        self::assertEmpty($instance->additional_user_props);
        $instance->setAdditionalUserProps($value);
        self::assertEquals($value, is_array($value) ? $instance->getAdditionalUserProps()->toArray() : $instance->getAdditionalUserProps());
        self::assertEquals($value, is_array($value) ? $instance->additional_user_props->toArray() : $instance->additional_user_props);
        if (!empty($value)) {
            self::assertNotNull($instance->getAdditionalUserProps());
            self::assertNotNull($instance->additional_user_props);
        }
    }

    /**
     * Test invalid property "additional_user_props"
     * @dataProvider invalidAdditionalUserPropsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAdditionalUserProps(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAdditionalUserProps($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAdditionalUserPropsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_additional_user_props'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAdditionalUserPropsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_additional_user_props'));
    }

    /**
     * Test property "receipt_industry_details"
     * @dataProvider validReceiptIndustryDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReceiptIndustryDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReceiptIndustryDetails());
        self::assertEmpty($instance->receipt_industry_details);
        self::assertIsObject($instance->getReceiptIndustryDetails());
        self::assertIsObject($instance->receipt_industry_details);
        self::assertCount(0, $instance->getReceiptIndustryDetails());
        self::assertCount(0, $instance->receipt_industry_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getReceiptIndustryDetails());
            self::assertNotNull($instance->receipt_industry_details);
            foreach ($value as $element) {
                if (is_object($element) && !empty($element)) {
                    $instance->addReceiptIndustryDetails($element);
                    self::assertEquals($element, $instance->getReceiptIndustryDetails()->get(0));
                    self::assertIsObject($instance->getReceiptIndustryDetails()->get(0));
                    self::assertIsObject($instance->receipt_industry_details->get(0));
                    self::assertIsObject($instance->getReceiptIndustryDetails());
                    self::assertIsObject($instance->receipt_industry_details);
                    $instance->getReceiptIndustryDetails()->clear();
                }
            }

            $instance->setReceiptIndustryDetails($value);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getReceiptIndustryDetails()[$key]->toArray());
                    self::assertEquals($element, $instance->receipt_industry_details[$key]->toArray());
                    self::assertIsArray($instance->getReceiptIndustryDetails()[$key]->toArray());
                    self::assertIsArray($instance->receipt_industry_details[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getReceiptIndustryDetails()->get($key));
                    self::assertIsObject($instance->getReceiptIndustryDetails()->get($key));
                    self::assertIsObject($instance->receipt_industry_details->get($key));
                    self::assertIsObject($instance->getReceiptIndustryDetails());
                    self::assertIsObject($instance->receipt_industry_details);
                }
            }
            self::assertCount(count($value), $instance->getReceiptIndustryDetails());
            self::assertCount(count($value), $instance->receipt_industry_details);
        }
    }

    /**
     * Test invalid property "receipt_industry_details"
     * @dataProvider invalidReceiptIndustryDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReceiptIndustryDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReceiptIndustryDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReceiptIndustryDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_industry_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReceiptIndustryDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_industry_details'));
    }

    /**
     * Test property "receipt_operational_details"
     * @dataProvider validReceiptOperationalDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReceiptOperationalDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReceiptOperationalDetails());
        self::assertEmpty($instance->receipt_operational_details);
        $instance->setReceiptOperationalDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getReceiptOperationalDetails()->toArray() : $instance->getReceiptOperationalDetails());
        self::assertEquals($value, is_array($value) ? $instance->receipt_operational_details->toArray() : $instance->receipt_operational_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getReceiptOperationalDetails());
            self::assertNotNull($instance->receipt_operational_details);
        }
    }

    /**
     * Test invalid property "receipt_operational_details"
     * @dataProvider invalidReceiptOperationalDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReceiptOperationalDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReceiptOperationalDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReceiptOperationalDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_operational_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReceiptOperationalDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_operational_details'));
    }

    /**
     * @dataProvider validNormalizationDataProvider
     *
     * @param mixed $withShipping
     * @param mixed $items
     * @param mixed $amount
     * @param mixed $expected
     */
    public function testNormalize($items, $amount, $expected, $withShipping = false): void
    {
        $receipt = new Receipt();
        foreach ($items as $itemInfo) {
            $item = new ReceiptItem();
            $item->setPrice(new ReceiptItemAmount($itemInfo['price']));
            if (!empty($itemInfo['quantity'])) {
                $item->setQuantity($itemInfo['quantity']);
            } else {
                $item->setQuantity(1);
            }
            if (!empty($itemInfo['shipping'])) {
                $item->setIsShipping(true);
            }
            $receipt->addItem($item);
        }
        $receipt->normalize(new ReceiptItemAmount($amount), $withShipping);

        self::assertEquals(count($expected), count($receipt->getItems()));
        $expectedAmount = 0;
        foreach ($receipt->getItems() as $index => $item) {
            self::assertEquals($expected[$index]['price'], $item->getPrice()->getIntegerValue());
            self::assertEquals($expected[$index]['quantity'], $item->getQuantity());

            $expectedAmount += $item->getAmount();
        }
        self::assertEquals($expectedAmount, $amount * 100.0);
    }

    public static function validNormalizationDataProvider()
    {
        return [
            [
                [
                    ['price' => 10.0],
                ],
                9.0,
                [
                    ['price' => 900, 'quantity' => 1.0],
                ]
            ],
            [
                [
                    ['price' => 10.0],
                    ['price' => 20.0],
                ],
                29.0,
                [
                    ['price' => 967, 'quantity' => 1.0],
                    ['price' => 1933, 'quantity' => 1.0],
                ]
            ],
            [
                [
                    ['price' => 10.0, 'quantity' => 1],
                    ['price' => 20.0, 'quantity' => 3],
                ],
                29.0,
                [
                    ['price' => 413, 'quantity' => 1.0],
                    ['price' => 829, 'quantity' => 3.0],
                ]
            ],
            [
                [
                    ['price' => 50.0, 'quantity' => 3],
                    ['price' => 20.0, 'quantity' => 3],
                ],
                100.0,
                [
                    ['price' => 2381, 'quantity' => 2.0],
                    ['price' => 2382, 'quantity' => 1.0],
                    ['price' => 952, 'quantity' => 3.0],
                ]
            ],
            [
                [
                    ['price' => 10.0, 'shipping' => true],
                    ['price' => 50.0, 'quantity' => 3],
                    ['price' => 10.0, 'shipping' => true],
                    ['price' => 20.0, 'quantity' => 3],
                ],
                120.0,
                [
                    ['price' => 1000, 'quantity' => 1.0],
                    ['price' => 2381, 'quantity' => 2.0],
                    ['price' => 2382, 'quantity' => 1.0],
                    ['price' => 1000, 'quantity' => 1.0],
                    ['price' => 952, 'quantity' => 3.0],
                ]
            ],
            [
                [
                    ['price' => 50.0, 'quantity' => 1, 'shipping' => 1],
                    ['price' => 50.0, 'quantity' => 2],
                    ['price' => 20.0, 'quantity' => 3],
                ],
                100.0,
                [
                    ['price' => 2381, 'quantity' => 1.0],
                    ['price' => 2381, 'quantity' => 1.0],
                    ['price' => 2382, 'quantity' => 1.0],
                    ['price' => 952, 'quantity' => 3.0],
                ],
                true
            ],
            [
                [
                    ['price' => 50.0, 'quantity' => 1, 'shipping' => 1],
                ],
                49.0,
                [
                    ['price' => 4900, 'quantity' => 1.0],
                ],
                true
            ],
            [
                [
                    ['price' => 100.0, 'quantity' => 0.5],
                    ['price' => 100.0, 'quantity' => 0.4],
                ],
                98.0,
                [
                    ['price' => 10889, 'quantity' => 0.25],
                    ['price' => 10888, 'quantity' => 0.25],
                    ['price' => 10889, 'quantity' => 0.4],
                ],
                true
            ],
            [
                [
                    ['price' => 10, 'quantity' => 1],
                    ['price' => 300, 'quantity' => 1, 'shipping' => 1],
                ],
                10.0,
                [
                    ['price' => 32, 'quantity' => 1],
                    ['price' => 968, 'quantity' => 1, 'shipping' => 1],
                ],
                true
            ],
            [
                [
                    ['price' => 10, 'quantity' => 1],
                    ['price' => 300, 'quantity' => 1, 'shipping' => 1],
                ],
                10.0,
                [
                    ['price' => 32, 'quantity' => 1],
                    ['price' => 968, 'quantity' => 1, 'shipping' => 1],
                ],
                false
            ],
            [
                [
                    ['price' => 0.01, 'quantity' => 1],
                    ['price' => 0.02, 'quantity' => 1],
                    ['price' => 0.03, 'quantity' => 1],
                    ['price' => 300, 'quantity' => 1, 'shipping' => 1],
                ],
                0.06,
                [
                    ['price' => 1, 'quantity' => 1],
                    ['price' => 1, 'quantity' => 1],
                    ['price' => 1, 'quantity' => 1],
                    ['price' => 3, 'quantity' => 1, 'shipping' => 1],
                ],
                false
            ],
            [
                [
                    ['price' => 0.01, 'quantity' => 7],
                    ['price' => 0.02, 'quantity' => 11],
                    ['price' => 0.03, 'quantity' => 13],
                    ['price' => 300, 'quantity' => 1, 'shipping' => 1],
                ],
                0.60,
                [
                    ['price' => 1, 'quantity' => 7],
                    ['price' => 1, 'quantity' => 11],
                    ['price' => 1, 'quantity' => 13],
                    ['price' => 29, 'quantity' => 1, 'shipping' => 1],
                ],
                false
            ],
            [
                [
                    ['price' => 0.01, 'quantity' => 7],
                    ['price' => 0.02, 'quantity' => 11],
                    ['price' => 10, 'quantity' => 1],
                    ['price' => 300, 'quantity' => 1, 'shipping' => 1],
                ],
                10.29,
                [
                    ['price' => 1, 'quantity' => 7],
                    ['price' => 1, 'quantity' => 11],
                    ['price' => 33, 'quantity' => 1],
                    ['price' => 978, 'quantity' => 1, 'shipping' => 1],
                ],
                false
            ],
        ];
    }

    /**
     * @dataProvider fromArrayDataProvider
     * @param array $source
     * @param array $expected
     */
    public function testFromArray(array $source, array $expected): void
    {
        $receipt = new Receipt($source);

        if (!empty($expected)) {
            foreach ($expected as $property => $value) {
                $propertyValue = $receipt->offsetGet($property);
                if ($propertyValue instanceof ListObject) {
                    self::assertEquals($value, $propertyValue->getItems()->toArray());
                } else {
                    self::assertEquals($value, $propertyValue);
                }
            }
        } else {
            self::assertEquals(array(), $receipt->getItems()->toArray());
            self::assertEquals(array(), $receipt->getSettlements()->toArray());
        }
    }

    public function fromArrayDataProvider(): array
    {
        $receiptItem = new ReceiptItem();
        $receiptItem->setDescription('test');
        $receiptItem->setQuantity(322);
        $receiptItem->setVatCode(4);
        $receiptItem->setPrice(new ReceiptItemAmount(5, 'USD'));

        $settlement = new Settlement();
        $settlement->setType(SettlementType::PREPAYMENT);
        $settlement->setAmount(new MonetaryAmount(123, 'USD'));

        return [
            [
                [],
                [],
            ],

            [
                [
                    'description' => Random::str(2, 128),
                    'taxSystemCode' => 2,
                    'customer' => [
                        'phone' => '1234567890',
                        'email' => 'test@tset.ru',
                    ],
                    'items' => [
                        [
                            'description' => 'test',
                            'amount' => [
                                'value' => 5,
                                'currency' => CurrencyCode::USD,
                            ],
                            'quantity' => 322,
                            'vat_code' => 4,
                        ],
                    ],
                    'settlements' => [
                        [
                            'type' => SettlementType::PREPAYMENT,
                            'amount' => [
                                'value' => 123,
                                'currency' => CurrencyCode::USD,
                            ],
                        ]
                    ]
                ],
                [
                    'tax_system_code' => 2,
                    'customer' => new ReceiptCustomer([
                        'phone' => '1234567890',
                        'email' => 'test@tset.ru',
                    ]),
                    'items' => [
                        $receiptItem,
                    ],
                    'settlements' => [
                        $settlement,
                    ]
                ],
            ],

            [
                [
                    'tax_system_code' => 3,
                    'customer' => [
                        'phone' => '1234567890',
                        'email' => 'test@tset.com',
                    ],
                    'items' => [
                        [
                            'description' => 'test',
                            'quantity' => 322,
                            'amount' => [
                                'value' => 5,
                                'currency' => 'USD',
                            ],
                            'vat_code' => 4,
                        ],
                        [
                            'description' => 'test',
                            'quantity' => 322,
                            'amount' => new ReceiptItemAmount(5, 'USD'),
                            'vat_code' => 4,
                        ],
                        [
                            'description' => 'test',
                            'quantity' => 322,
                            'amount' => new ReceiptItemAmount([
                                'value' => 5,
                                'currency' => 'USD',
                            ]),
                            'vat_code' => 4,
                        ],
                    ],
                    'settlements' => [
                        [
                            'type' => SettlementType::PREPAYMENT,
                            'amount' => [
                                'value' => 123,
                                'currency' => 'USD'
                            ]
                        ],
                        [
                            'type' => SettlementType::PREPAYMENT,
                            'amount' => [
                                'value' => 123,
                                'currency' => 'USD'
                            ]
                        ]
                    ],
                    'receipt_operational_details' => [
                        'operation_id' => 255,
                        'value' => '00-tr-589',
                        'created_at' => '2012-11-03T11:52:31.827Z',
                    ],
                ],
                [
                    'taxSystemCode' => 3,
                    'customer' => new ReceiptCustomer([
                        'phone' => '1234567890',
                        'email' => 'test@tset.com',
                    ]),
                    'items' => [
                        $receiptItem,
                        $receiptItem,
                        $receiptItem,
                    ],
                    'settlements' => [
                        $settlement,
                        $settlement
                    ],
                    'receipt_operational_details' => new OperationalDetails([
                        'operation_id' => 255,
                        'value' => '00-tr-589',
                        'created_at' => '2012-11-03T11:52:31.827Z',
                    ]),
                ],
            ],
        ];
    }
}
