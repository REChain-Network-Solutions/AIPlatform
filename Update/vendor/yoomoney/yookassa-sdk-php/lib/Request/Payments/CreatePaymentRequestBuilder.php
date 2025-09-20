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

namespace YooKassa\Request\Payments;

use Exception;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\Deal\PaymentDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesFactory;
use YooKassa\Request\Payments\PaymentData\AbstractPaymentData;
use YooKassa\Request\Payments\PaymentData\PaymentDataFactory;
use YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder;
use YooKassa\Request\Payments\ReceiverData\AbstractReceiver;

/**
 * Класс, представляющий модель CreatePaymentRequestBuilder.
 *
 * Класс билдера объекта запроса на создание платежа, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 02-builder.php 11 75 Пример использования билдера
 */
class CreatePaymentRequestBuilder extends AbstractPaymentRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreatePaymentRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * @var Recipient|null Получатель платежа
     */
    private ?Recipient $recipient = null;

    /**
     * @var PaymentDataFactory|null Фабрика методов проведения платежей
     */
    private ?PaymentDataFactory $paymentDataFactory = null;

    /**
     * @var ConfirmationAttributesFactory|null Фабрика объектов методов подтверждения платежей
     */
    private ?ConfirmationAttributesFactory $confirmationFactory = null;

    /**
     * Устанавливает идентификатор магазина получателя платежа.
     *
     * @deprecated Больше не используется
     *
     * @param string $value Идентификатор магазина
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws EmptyPropertyValueException Выбрасывается если было передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не строковое значение
     * @throws Exception
     */
    public function setAccountId(string $value): CreatePaymentRequestBuilder
    {
        return $this;
    }

    /**
     * Устанавливает идентификатор шлюза.
     *
     * @param string $value Идентификатор шлюза
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws EmptyPropertyValueException Выбрасывается если было передано пустое значение
     * @throws InvalidPropertyValueTypeException Выбрасывается если было передано не строковое значение
     */
    public function setGatewayId(string $value): CreatePaymentRequestBuilder
    {
        $this->recipient->setGatewayId($value);

        return $this;
    }

    /**
     * Устанавливает получателя платежа из объекта или ассоциативного массива.
     *
     * @param array|Recipient|null $value Получатель платежа
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если передан аргумент не валидного типа
     */
    public function setRecipient(mixed $value): CreatePaymentRequestBuilder
    {
        if (is_array($value)) {
            $this->recipient->fromArray($value);
        } elseif ($value instanceof Recipient) {
            $this->recipient->setGatewayId($value->getGatewayId());
        } else {
            throw new InvalidPropertyValueTypeException('Invalid recipient value', 0, 'recipient', $value);
        }

        return $this;
    }

    /**
     * Устанавливает информацию об авиабилетах.
     *
     * @param AirlineInterface|array|null $value Объект данных длинной записи или ассоциативный массив с данными
     */
    public function setAirline(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setAirline($value);

        return $this;
    }

    /**
     * Устанавливает одноразовый токен для проведения оплаты.
     *
     * @param string|null $value Одноразовый токен для проведения оплаты
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение превышает допустимую длину
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является строкой
     */
    public function setPaymentToken(?string $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setPaymentToken($value);

        return $this;
    }

    /**
     * Устанавливает идентификатор записи о сохранённых данных покупателя.
     *
     * @param string|null $value Идентификатор записи о сохраненных платежных данных покупателя
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     */
    public function setPaymentMethodId(?string $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setPaymentMethodId($value);

        return $this;
    }

    /**
     * Устанавливает объект с информацией для создания метода оплаты.
     *
     * @param null|AbstractPaymentData|array|string $value Объект создания метода оплаты или null
     * @param array|null $options Настройки способа оплаты в виде ассоциативного массива
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если был передан объект невалидного типа
     */
    public function setPaymentMethodData(mixed $value, ?array $options = null): CreatePaymentRequestBuilder
    {
        if (is_string($value) && '' !== $value) {
            if (empty($options)) {
                $value = $this->getPaymentDataFactory()->factory($value);
            } else {
                $value = $this->getPaymentDataFactory()->factoryFromArray($options, $value);
            }
        } elseif (is_array($value)) {
            $value = $this->getPaymentDataFactory()->factoryFromArray($value);
        }
        $this->currentObject->setPaymentMethodData($value);

        return $this;
    }

    /**
     * Устанавливает способ подтверждения платежа.
     *
     * @param null|AbstractConfirmationAttributes|array|string $value Способ подтверждения платежа
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является объектом типа
     *                                           AbstractConfirmationAttributes или null
     */
    public function setConfirmation(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setConfirmation($value);

        return $this;
    }

    /**
     * Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method.
     *
     * @param bool|null $value Сохранить платежные данные для последующего использования
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Генерируется если переданный аргумент не кастится в bool
     */
    public function setSavePaymentMethod(?bool $value = null): CreatePaymentRequestBuilder
    {
        $this->currentObject->setSavePaymentMethod($value);

        return $this;
    }

    /**
     * Устанавливает флаг автоматического принятия поступившей оплаты.
     *
     * @param bool $value Автоматически принять поступившую оплату
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Генерируется если переданный аргумент не кастится в bool
     */
    public function setCapture(bool $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setCapture($value);

        return $this;
    }

    /**
     * Устанавливает IP адрес покупателя.
     *
     * @param string|null $value IPv4 или IPv6-адрес покупателя
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданный аргумент не является строкой
     */
    public function setClientIp(?string $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setClientIp($value);

        return $this;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $value Метаданные платежа, устанавливаемые мерчантом
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданные данные не удалось интерпретировать как
     *                                           метаданные платежа
     */
    public function setMetadata(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setMetadata($value);

        return $this;
    }

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $value Описание транзакции
     *
     * @return CreatePaymentRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение превышает допустимую длину
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является строкой
     */
    public function setDescription(?string $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setDescription($value);

        return $this;
    }

    /**
     * Устанавливает сделку.
     *
     * @param null|array|PaymentDealInfo $value Данные о сделке, в составе которой проходит платеж
     *
     * @return CreatePaymentRequestBuilder Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException
     */
    public function setDeal(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setDeal($value);

        return $this;
    }

    /**
     * Устанавливает информацию для проверки операции на мошенничество.
     * @deprecated Больше не поддерживается. Вместо него нужно использовать `setReceiver()`
     *
     * @param null|array|FraudData $value Информация для проверки операции на мошенничество
     *
     * @return CreatePaymentRequestBuilder Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException
     */
    public function setFraudData(mixed $value): CreatePaymentRequestBuilder
    {
        return $this;
    }

    /**
     * Устанавливает идентификатор покупателя в вашей системе.
     *
     * @param string|null $value Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданный аргумент не является строкой
     */
    public function setMerchantCustomerId(?string $value): self
    {
        $this->currentObject->setMerchantCustomerId($value);

        return $this;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|IndustryDetails[]|null $value Отраслевой реквизит чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptIndustryDetails(mixed $value): self
    {
        $this->receipt->setReceiptIndustryDetails($value);

        return $this;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|IndustryDetails[]|null $value Отраслевой реквизит чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptOperationalDetails(mixed $value): self
    {
        $this->receipt->setReceiptOperationalDetails($value);

        return $this;
    }

    /**
     * Устанавливает реквизиты получателя оплаты.
     *
     * @param null|array|AbstractReceiver $value Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона
     *
     * @return CreatePaymentRequestBuilder Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException
     */
    public function setReceiver(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setReceiver($value);

        return $this;
    }

    /**
     * Устанавливает платежное поручение.
     *
     * @param null|array|AbstractPaymentOrder $value Платежное поручение
     *
     * @return CreatePaymentRequestBuilder Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException
     */
    public function setPaymentOrder(mixed $value): CreatePaymentRequestBuilder
    {
        $this->currentObject->setPaymentOrder($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return CreatePaymentRequestInterface|AbstractRequestInterface Инстанс объекта запроса
     *
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }
        $gatewayId = $this->recipient->getGatewayId();
        if (!empty($gatewayId)) {
            $this->currentObject->setRecipient($this->recipient);
        }
        if ($this->receipt->notEmpty()) {
            $this->currentObject->setReceipt($this->receipt);
        }

        $this->currentObject->setAmount($this->amount);

        return parent::build();
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreatePaymentRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): CreatePaymentRequest
    {
        parent::initCurrentObject();

        $request = new CreatePaymentRequest();

        $this->recipient = new Recipient();

        return $request;
    }

    /**
     * Возвращает фабрику методов проведения платежей.
     *
     * @return PaymentDataFactory Фабрика методов проведения платежей
     */
    protected function getPaymentDataFactory(): PaymentDataFactory
    {
        if (null === $this->paymentDataFactory) {
            $this->paymentDataFactory = new PaymentDataFactory();
        }

        return $this->paymentDataFactory;
    }
}
