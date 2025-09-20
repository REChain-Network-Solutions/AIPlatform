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

namespace Tests\YooKassa\Request\PersonalData\PersonalDataType;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\PersonalData\CreateSbpPayoutRecipientPersonalDataRequestBuilder;
use YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest;

/**
 * SbpPayoutRecipientPersonalDataRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class SbpPayoutRecipientPersonalDataRequestTest extends AbstractTestCase
{
    protected SbpPayoutRecipientPersonalDataRequest $object;

    /**
    * @return SbpPayoutRecipientPersonalDataRequest
    */
    protected function getTestInstance(): SbpPayoutRecipientPersonalDataRequest
    {
        return new SbpPayoutRecipientPersonalDataRequest();
    }

    /**
    * @return void
    */
    public function testSbpPayoutRecipientPersonalDataRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(SbpPayoutRecipientPersonalDataRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SbpPayoutRecipientPersonalDataRequest::class));
        $this->assertInstanceOf(SbpPayoutRecipientPersonalDataRequest::class, $this->object);
    }

    /**
    * Test property "type"
    *
    * @return void
    * @throws Exception
    */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertContains($instance->getType(), ['sbp_payout_recipient']);
        self::assertContains($instance->type, ['sbp_payout_recipient']);
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
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
    * Test property "last_name"
    * @dataProvider validLastNameDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testLastName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLastName($value);
        self::assertNotNull($instance->getLastName());
        self::assertNotNull($instance->last_name);
        self::assertEquals($value, is_array($value) ? $instance->getLastName()->toArray() : $instance->getLastName());
        self::assertEquals($value, is_array($value) ? $instance->last_name->toArray() : $instance->last_name);
        self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->getLastName());
        self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->last_name);
        self::assertLessThanOrEqual(200, is_string($instance->getLastName()) ? mb_strlen($instance->getLastName()) : $instance->getLastName());
        self::assertLessThanOrEqual(200, is_string($instance->last_name) ? mb_strlen($instance->last_name) : $instance->last_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getLastName()) ? mb_strlen($instance->getLastName()) : $instance->getLastName());
        self::assertGreaterThanOrEqual(1, is_string($instance->last_name) ? mb_strlen($instance->last_name) : $instance->last_name);
    }

    /**
    * Test invalid property "last_name"
    * @dataProvider invalidLastNameDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidLastName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setLastName($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validLastNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_last_name'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidLastNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_last_name'));
    }

    /**
    * Test property "first_name"
    * @dataProvider validFirstNameDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testFirstName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setFirstName($value);
        self::assertNotNull($instance->getFirstName());
        self::assertNotNull($instance->first_name);
        self::assertEquals($value, is_array($value) ? $instance->getFirstName()->toArray() : $instance->getFirstName());
        self::assertEquals($value, is_array($value) ? $instance->first_name->toArray() : $instance->first_name);
        self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->getFirstName());
        self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->first_name);
        self::assertLessThanOrEqual(100, is_string($instance->getFirstName()) ? mb_strlen($instance->getFirstName()) : $instance->getFirstName());
        self::assertLessThanOrEqual(100, is_string($instance->first_name) ? mb_strlen($instance->first_name) : $instance->first_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getFirstName()) ? mb_strlen($instance->getFirstName()) : $instance->getFirstName());
        self::assertGreaterThanOrEqual(1, is_string($instance->first_name) ? mb_strlen($instance->first_name) : $instance->first_name);
    }

    /**
    * Test invalid property "first_name"
    * @dataProvider invalidFirstNameDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidFirstName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFirstName($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validFirstNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_first_name'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidFirstNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_first_name'));
    }

    /**
    * Test property "middle_name"
    * @dataProvider validMiddleNameDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testMiddleName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMiddleName());
        self::assertEmpty($instance->middle_name);
        $instance->setMiddleName($value);
        self::assertEquals($value, is_array($value) ? $instance->getMiddleName()->toArray() : $instance->getMiddleName());
        self::assertEquals($value, is_array($value) ? $instance->middle_name->toArray() : $instance->middle_name);
        if (!empty($value)) {
            self::assertTrue($instance->hasMiddleName());
            self::assertNotNull($instance->getMiddleName());
            self::assertNotNull($instance->middle_name);
            self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->getMiddleName());
            self::assertMatchesRegularExpression("/^[—–‐\\-a-zA-Zа-яёА-ЯЁ ]*$/u", $instance->middle_name);
            self::assertLessThanOrEqual(200, is_string($instance->getMiddleName()) ? mb_strlen($instance->getMiddleName()) : $instance->getMiddleName());
            self::assertLessThanOrEqual(200, is_string($instance->middle_name) ? mb_strlen($instance->middle_name) : $instance->middle_name);
            self::assertGreaterThanOrEqual(1, is_string($instance->getMiddleName()) ? mb_strlen($instance->getMiddleName()) : $instance->getMiddleName());
            self::assertGreaterThanOrEqual(1, is_string($instance->middle_name) ? mb_strlen($instance->middle_name) : $instance->middle_name);
        }
    }

    /**
    * Test invalid property "middle_name"
    * @dataProvider invalidMiddleNameDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidMiddleName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMiddleName($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validMiddleNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_middle_name'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidMiddleNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_middle_name'));
    }

    /**
     * Test property "metadata"
     * @dataProvider validMetadataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMetadata(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMetadata());
        self::assertEmpty($instance->metadata);
        $instance->setMetadata($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasMetadata());
            self::assertTrue($instance->hasMetadata());
            self::assertNotNull($instance->getMetadata());
            self::assertNotNull($instance->metadata);
            foreach ($value as $key => $element) {
                if (!empty($element)) {
                    self::assertEquals($element, $instance->getMetadata()[$key]);
                    self::assertEquals($element, $instance->metadata[$key]);
                    self::assertIsObject($instance->getMetadata());
                    self::assertIsObject($instance->metadata);
                }
            }
            self::assertCount(count($value), $instance->getMetadata());
            self::assertCount(count($value), $instance->metadata);
            if ($instance->getMetadata() instanceof Metadata) {
                self::assertEquals($value, $instance->getMetadata()->toArray());
                self::assertEquals($value, $instance->metadata->toArray());
                self::assertCount(count($value), $instance->getMetadata());
                self::assertCount(count($value), $instance->metadata);
            }
        }
    }

    /**
     * Test invalid property "metadata"
     * @dataProvider invalidMetadataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMetadata(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMetadata($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * Test valid method "validate"
     *
     * @return void
     */
    public function testValidate(): void
    {
        $instance = new SbpPayoutRecipientPersonalDataRequest();
        self::assertFalse($instance->validate());

        $instance->setLastName('dfbs');
        self::assertFalse($instance->validate());

        $instance = new SbpPayoutRecipientPersonalDataRequest();
        $instance->setFirstName('test');
        self::assertFalse($instance->validate());
        $instance->setFirstName('dfbs');
        $instance->setLastName('test');
        self::assertTrue($instance->validate());
    }

    /**
     * Test valid method "builder"
     *
     * @return void
     */
    public function testBuilder(): void
    {
        $builder = SbpPayoutRecipientPersonalDataRequest::builder();
        self::assertInstanceOf(CreateSbpPayoutRecipientPersonalDataRequestBuilder::class, $builder);
    }
}
