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
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\ListObject;
use YooKassa\Helpers\Random;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\ReceiptCustomer;
use YooKassa\Model\Receipt\Settlement;
use YooKassa\Request\Receipts\CreatePostReceiptRequest;
use YooKassa\Request\Receipts\CreatePostReceiptRequestBuilder;

/**
 * CreatePostReceiptRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePostReceiptRequestTest extends AbstractTestCase
{
    protected CreatePostReceiptRequest $object;

    /**
     * @return CreatePostReceiptRequest
     */
    protected function getTestInstance(): CreatePostReceiptRequest
    {
        return new CreatePostReceiptRequest();
    }

    /**
     * @return void
     */
    public function testCreatePostReceiptRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(CreatePostReceiptRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CreatePostReceiptRequest::class));
        $this->assertInstanceOf(CreatePostReceiptRequest::class, $this->object);
    }

    /**
     * Test property "type"
     * @dataProvider validTypeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setType($value);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertEquals($value, is_array($value) ? $instance->getType()->toArray() : $instance->getType());
        self::assertEquals($value, is_array($value) ? $instance->type->toArray() : $instance->type);
    }

    /**
     * Test invalid property "type"
     * @dataProvider invalidTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
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
        $instance->setCustomer($value);
        self::assertTrue($instance->hasCustomer());
        self::assertNotNull($instance->getCustomer());
        self::assertNotNull($instance->customer);
        self::assertEquals($value, is_array($value) ? $instance->getCustomer()->toArray() : $instance->getCustomer());
        self::assertEquals($value, is_array($value) ? $instance->customer->toArray() : $instance->customer);
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
        self::assertFalse($instance->notEmpty());
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
     * Test property "send"
     * @dataProvider validSendDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSend(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setSend($value);
        self::assertNotNull($instance->getSend());
        self::assertNotNull($instance->send);
        self::assertEquals($value, is_array($value) ? $instance->getSend()->toArray() : $instance->getSend());
        self::assertEquals($value, is_array($value) ? $instance->send->toArray() : $instance->send);
        self::assertIsBool($instance->getSend());
        self::assertIsBool($instance->send);
    }

    /**
     * Test invalid property "send"
     * @dataProvider invalidSendDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSend(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSend($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSendDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_send'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSendDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_send'));
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
        $instance->setReceiptIndustryDetails($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getReceiptIndustryDetails());
            self::assertNotNull($instance->receipt_industry_details);
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
        self::assertIsObject($instance->getSettlements());
        self::assertIsObject($instance->settlements);
        self::assertCount(0, $instance->getSettlements());
        self::assertCount(0, $instance->settlements);
        $instance->setSettlements($value);
        self::assertNotNull($instance->getSettlements());
        self::assertNotNull($instance->settlements);
        self::assertFalse($instance->notEmpty());
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
     * Test property "on_behalf_of"
     * @dataProvider validOnBehalfOfDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testOnBehalfOf(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getOnBehalfOf());
        self::assertEmpty($instance->on_behalf_of);
        $instance->setOnBehalfOf($value);
        self::assertEquals($value, is_array($value) ? $instance->getOnBehalfOf()->toArray() : $instance->getOnBehalfOf());
        self::assertEquals($value, is_array($value) ? $instance->on_behalf_of->toArray() : $instance->on_behalf_of);
        if (!empty($value)) {
            self::assertNotNull($instance->getOnBehalfOf());
            self::assertNotNull($instance->on_behalf_of);
        }
    }

    /**
     * Test invalid property "on_behalf_of"
     * @dataProvider invalidOnBehalfOfDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidOnBehalfOf(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setOnBehalfOf($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validOnBehalfOfDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_on_behalf_of'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidOnBehalfOfDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_on_behalf_of'));
    }

    /**
     * Test valid method "validate"
     *
     * @return void
     * @throws Exception
     */
    public function testValidate(): void
    {
        $instance = new CreatePostReceiptRequest();
        self::assertFalse($instance->validate());

        $instance->setCustomer($this->validCustomerDataProvider());
        self::assertFalse($instance->validate());

        $instance->setType('simple');
        self::assertFalse($instance->validate());

        $instance->setObjectType(null);
        self::assertFalse($instance->validate());

        $instance->setObjectType('simple');
        self::assertFalse($instance->validate());
        self::assertEquals('simple', $instance->getObjectType());

        $instance->setObjectId(1);
        self::assertFalse($instance->validate());
        self::assertEquals(1, $instance->getObjectId());

        $settlements = $this->validSettlementsDataProvider();
        $instance->setSettlements($settlements[0][0]);
        self::assertFalse($instance->validate());

        $items = $this->validItemsDataProvider();
        $instance->setItems($items[0][0]);
        self::assertTrue($instance->validate());
    }

    public function testBuilder(): void
    {
        $builder = CreatePostReceiptRequest::builder();
        self::assertInstanceOf(CreatePostReceiptRequestBuilder::class, $builder);
    }
}
