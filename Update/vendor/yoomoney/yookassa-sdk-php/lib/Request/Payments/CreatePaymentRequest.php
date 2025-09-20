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

use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\PaymentDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Payment;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes;
use YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesFactory;
use YooKassa\Request\Payments\PaymentData\AbstractPaymentData;
use YooKassa\Request\Payments\PaymentData\PaymentDataFactory;
use YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder;
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderFactory;
use YooKassa\Request\Payments\ReceiverData\AbstractReceiver;
use YooKassa\Request\Payments\ReceiverData\ReceiverFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreateCaptureRequest.
 *
 * Класс объекта запроса к API на проведение нового платежа.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 02-builder.php 11 75 Пример использования билдера
 *
 * @property Recipient $recipient Получатель платежа, если задан
 * @property AmountInterface $amount Сумма создаваемого платежа
 * @property string $description Описание транзакции
 * @property ReceiptInterface $receipt Данные фискального чека 54-ФЗ
 * @property string $paymentToken Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
 * @property string $payment_token Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
 * @property string $paymentMethodId Идентификатор записи о сохраненных платежных данных покупателя
 * @property string $payment_method_id Идентификатор записи о сохраненных платежных данных покупателя
 * @property AbstractPaymentData $paymentMethodData Данные используемые для создания метода оплаты
 * @property AbstractPaymentData $payment_method_data Данные используемые для создания метода оплаты
 * @property AbstractConfirmationAttributes $confirmation Способ подтверждения платежа
 * @property bool|null $savePaymentMethod Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method
 * @property bool|null $save_payment_method Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method
 * @property bool $capture Автоматически принять поступившую оплату
 * @property string $clientIp IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения
 * @property string $client_ip IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения
 * @property Metadata $metadata Метаданные привязанные к платежу
 * @property PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
 * @property string $merchantCustomerId Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property string $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property AbstractPaymentOrder $payment_order Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ. Необходимо передавать при [оплате ЖКУ](/developers/payment-acceptance/scenario-extensions/utility-payments).
 * @property AbstractReceiver|null $receiver Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона
 */
class CreatePaymentRequest extends AbstractPaymentRequest implements CreatePaymentRequestInterface
{
    public const MAX_LENGTH_PAYMENT_TOKEN = 10240;

    /**
     * @var Recipient|null Получатель платежа
     */
    #[Assert\Valid]
    #[Assert\Type(Recipient::class)]
    private ?Recipient $_recipient = null;

    /**
     * @var string|null Описание транзакции
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: Payment::MAX_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * @var string|null Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_PAYMENT_TOKEN)]
    private ?string $_payment_token = null;

    /**
     * @var string|null Идентификатор записи о сохраненных платежных данных покупателя
     */
    #[Assert\Type('string')]
    private ?string $_payment_method_id = null;

    /**
     * @var AbstractPaymentData|null Данные используемые для создания метода оплаты
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractPaymentData::class)]
    private ?AbstractPaymentData $_payment_method_data = null;

    /**
     * @var AbstractConfirmationAttributes|null Способ подтверждения платежа
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractConfirmationAttributes::class)]
    private ?AbstractConfirmationAttributes $_confirmation = null;

    /**
     * @var bool|null Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method.
     */
    #[Assert\Type('bool')]
    private ?bool $_save_payment_method = null;

    /**
     * @var bool Автоматически принять поступившую оплату
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private bool $_capture = true;

    /**
     * @var string|null IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения.
     */
    #[Assert\Type('string')]
    #[Assert\Ip(Assert\Ip::ALL)]
    private ?string $_client_ip = null;

    /**
     * @var Metadata|null Метаданные привязанные к платежу
     */
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * @var PaymentDealInfo|null Данные о сделке, в составе которой проходит платеж. Необходимо передавать, если вы проводите Безопасную сделку
     */
    #[Assert\Valid]
    #[Assert\Type(PaymentDealInfo::class)]
    private ?PaymentDealInfo $_deal = null;

    /**
     * @var FraudData|null Информация для проверки операции на мошенничество
     * @deprecated Больше не поддерживается. Вместо него нужно использовать `receiver`
     */
    #[Assert\Valid]
    #[Assert\Type(FraudData::class)]
    private ?FraudData $_fraud_data = null;

    /**
     * Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов.
     *
     * Присутствует, если вы хотите запомнить банковскую карту и отобразить ее при повторном платеже в виджете ЮKassa
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: Payment::MAX_LENGTH_MERCHANT_CUSTOMER_ID)]
    private ?string $_merchant_customer_id = null;

    /**
     * Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.
     *
     * Необходимо передавать при [оплате ЖКУ](/developers/payment-acceptance/scenario-extensions/utility-payments).
     *
     * @var AbstractPaymentOrder|null
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractPaymentOrder::class)]
    private ?AbstractPaymentOrder $_payment_order = null;

    /**
     * Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона
     *
     * @var AbstractReceiver|null
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractReceiver::class)]
    private ?AbstractReceiver $_receiver = null;

    /**
     * Возвращает описание транзакции
     * @return string|null
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает описание транзакции
     * @param string|null $description
     *
     * @return CreatePaymentRequest
     */
    public function setDescription(?string $description): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Проверяет наличие описания транзакции в создаваемом платеже.
     *
     * @return bool True если описание транзакции есть, false если нет
     */
    public function hasDescription(): bool
    {
        return !empty($this->_description);
    }

    /**
     * Возвращает объект получателя платежа.
     *
     * @return null|Recipient Объект с информацией о получателе платежа или null, если получатель не задан
     */
    public function getRecipient(): ?Recipient
    {
        return $this->_recipient;
    }

    /**
     * Проверяет наличие получателя платежа в запросе.
     *
     * @return bool True если получатель платежа задан, false если нет
     */
    public function hasRecipient(): bool
    {
        return !empty($this->_recipient);
    }

    /**
     * Устанавливает объект с информацией о получателе платежа.
     *
     * @param null|array|Recipient $recipient Инстанс объекта информации о получателе платежа или null
     */
    public function setRecipient(mixed $recipient): self
    {
        $this->_recipient = $this->validatePropertyValue('_recipient', $recipient);
        return $this;
    }

    /**
     * Возвращает одноразовый токен для проведения оплаты.
     *
     * @return string|null Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
     */
    public function getPaymentToken(): ?string
    {
        return $this->_payment_token;
    }

    /**
     * Проверяет наличие одноразового токена для проведения оплаты.
     *
     * @return bool True если токен установлен, false если нет
     */
    public function hasPaymentToken(): bool
    {
        return !empty($this->_payment_token);
    }

    /**
     * Устанавливает одноразовый токен для проведения оплаты, сформированный YooKassa JS widget.
     *
     * @param string|null $payment_token Одноразовый токен для проведения оплаты
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение превышает допустимую длину
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является строкой
     */
    public function setPaymentToken(?string $payment_token): self
    {
        $this->_payment_token = $this->validatePropertyValue('_payment_token', $payment_token);
        return $this;
    }

    /**
     * Устанавливает идентификатор записи платёжных данных покупателя.
     *
     * @return string|null Идентификатор записи о сохраненных платежных данных покупателя
     */
    public function getPaymentMethodId(): ?string
    {
        return $this->_payment_method_id;
    }

    /**
     * Проверяет наличие идентификатора записи о платёжных данных покупателя.
     *
     * @return bool True если идентификатор задан, false если нет
     */
    public function hasPaymentMethodId(): bool
    {
        return !empty($this->_payment_method_id);
    }

    /**
     * Устанавливает идентификатор записи о сохранённых данных покупателя.
     *
     * @param string|null $payment_method_id Идентификатор записи о сохраненных платежных данных покупателя
     *
     * @throws InvalidPropertyValueTypeException Генерируется если переданные значение не является строкой или null
     */
    public function setPaymentMethodId(?string $payment_method_id): self
    {
        $this->_payment_method_id = $this->validatePropertyValue('_payment_method_id', $payment_method_id);
        return $this;
    }

    /**
     * Возвращает данные для создания метода оплаты.
     *
     * @return AbstractPaymentData|null Данные используемые для создания метода оплаты
     */
    public function getPaymentMethodData(): ?AbstractPaymentData
    {
        return $this->_payment_method_data;
    }

    /**
     * Проверяет установлен ли объект с методом оплаты.
     *
     * @return bool True если объект метода оплаты установлен, false если нет
     */
    public function hasPaymentMethodData(): bool
    {
        return !empty($this->_payment_method_data);
    }

    /**
     * Устанавливает объект с информацией для создания метода оплаты.
     *
     * @param null|array|AbstractPaymentData $payment_method_data Объект создания метода оплаты или null
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если был передан объект невалидного типа
     */
    public function setPaymentMethodData(mixed $payment_method_data): self
    {
        if (is_array($payment_method_data)) {
            $payment_method_data = (new PaymentDataFactory())->factoryFromArray($payment_method_data);
        }
        $this->_payment_method_data = $this->validatePropertyValue('_payment_method_data', $payment_method_data);
        return $this;
    }

    /**
     * Возвращает способ подтверждения платежа.
     *
     * @return AbstractConfirmationAttributes|null Способ подтверждения платежа
     */
    public function getConfirmation(): ?AbstractConfirmationAttributes
    {
        return $this->_confirmation;
    }

    /**
     * Проверяет, был ли установлен способ подтверждения платежа.
     *
     * @return bool True если способ подтверждения платежа был установлен, false если нет
     */
    public function hasConfirmation(): bool
    {
        return null !== $this->_confirmation;
    }

    /**
     * Устанавливает способ подтверждения платежа.
     *
     * @param null|array|AbstractConfirmationAttributes $confirmation Способ подтверждения платежа
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является объектом типа
     *                                           AbstractConfirmationAttributes или null
     */
    public function setConfirmation(mixed $confirmation): self
    {
        if (is_array($confirmation)) {
            $factory = new ConfirmationAttributesFactory();
            $confirmation = $factory->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

    /**
     * Возвращает флаг сохранения платёжных данных.
     *
     * @return bool|null Флаг сохранения платёжных данных
     */
    public function getSavePaymentMethod(): ?bool
    {
        return $this->_save_payment_method;
    }

    /**
     * Проверяет, был ли установлен флаг сохранения платёжных данных.
     *
     * @return bool True если флаг был установлен, false если нет
     */
    public function hasSavePaymentMethod(): bool
    {
        return isset($this->_save_payment_method);
    }

    /**
     * Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method.
     *
     * @param bool|null $save_payment_method Сохранить платежные данные для последующего использования
     *
     * @throws InvalidPropertyValueTypeException Генерируется если переданный аргумент не кастится в bool
     */
    public function setSavePaymentMethod(?bool $save_payment_method = null): self
    {
        $this->_save_payment_method = $this->validatePropertyValue('_save_payment_method', $save_payment_method);
        return $this;
    }

    /**
     * Возвращает флаг автоматического принятия поступившей оплаты.
     *
     * @return bool True если требуется автоматически принять поступившую оплату, false если нет
     */
    public function getCapture(): bool
    {
        return $this->_capture;
    }

    /**
     * Проверяет, был ли установлен флаг автоматического принятия поступившей оплаты.
     *
     * @return bool True если флаг автоматического принятия оплаты был установлен, false если нет
     */
    public function hasCapture(): bool
    {
        return isset($this->_capture);
    }

    /**
     * Устанавливает флаг автоматического принятия поступившей оплаты.
     *
     * @param bool $capture Автоматически принять поступившую оплату
     *
     * @throws InvalidPropertyValueTypeException Генерируется если переданный аргумент не кастится в bool
     */
    public function setCapture(bool $capture): self
    {
        $this->_capture = $this->validatePropertyValue('_capture', $capture);
        return $this;
    }

    /**
     * Возвращает IPv4 или IPv6-адрес покупателя.
     *
     * @return string|null IPv4 или IPv6-адрес покупателя
     */
    public function getClientIp(): ?string
    {
        return $this->_client_ip;
    }

    /**
     * Проверяет, был ли установлен IPv4 или IPv6-адрес покупателя.
     *
     * @return bool True если IP адрес покупателя был установлен, false если нет
     */
    public function hasClientIp(): bool
    {
        return null !== $this->_client_ip;
    }

    /**
     * Устанавливает IP адрес покупателя.
     *
     * @param string|null $client_ip IPv4 или IPv6-адрес покупателя
     */
    public function setClientIp(?string $client_ip): self
    {
        $this->_client_ip = $this->validatePropertyValue('_client_ip', $client_ip);
        return $this;
    }

    /**
     * Возвращает данные оплаты установленные мерчантом
     *
     * @return Metadata|null Метаданные, привязанные к платежу
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Проверяет, были ли установлены метаданные заказа.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool
    {
        return !empty($this->_metadata) && $this->_metadata->count() > 0;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $metadata Метаданные платежа, устанавливаемые мерчантом
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданные данные не удалось интерпретировать как
     *                                           метаданные платежа
     */
    public function setMetadata(mixed $metadata): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

    /**
     * Возвращает данные о сделке, в составе которой проходит платеж.
     *
     * @return PaymentDealInfo|null Данные о сделке, в составе которой проходит платеж
     */
    public function getDeal(): ?PaymentDealInfo
    {
        return $this->_deal;
    }

    /**
     * Проверяет, были ли установлены данные о сделке.
     *
     * @return bool True если данные о сделке были установлены, false если нет
     */
    public function hasDeal(): bool
    {
        return !empty($this->_deal);
    }

    /**
     * Устанавливает данные о сделке, в составе которой проходит платеж.
     *
     * @param null|array|PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданные данные не удалось интерпретировать как метаданные платежа
     */
    public function setDeal(mixed $deal): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает информацию для проверки операции на мошенничество.
     *
     * @deprecated Больше не поддерживается. Вместо него нужно использовать `getReceiver()`
     * @return null|FraudData Информация для проверки операции на мошенничество
     */
    public function getFraudData(): ?FraudData
    {
        return null;
    }

    /**
     * Устанавливает информацию для проверки операции на мошенничество.
     *
     * @param null|array|FraudData $fraud_data Информация для проверки операции на мошенничество
     * @deprecated Больше не поддерживается. Вместо него нужно использовать `setReceiver()`
     */
    public function setFraudData(mixed $fraud_data = null): self
    {
        return $this;
    }

    /**
     * Проверяет, была ли установлена информация для проверки операции на мошенничество.
     *
     * @deprecated Больше не поддерживается. Вместо него нужно использовать `hasReceiver()`
     * @return bool True если информация была установлена, false если нет
     */
    public function hasFraudData(): bool
    {
        return false;
    }

    /**
     * Возвращает идентификатор покупателя в вашей системе.
     *
     * @return string|null Идентификатор покупателя в вашей системе
     */
    public function getMerchantCustomerId(): ?string
    {
        return $this->_merchant_customer_id;
    }

    /**
     * Проверяет, был ли установлен идентификатор покупателя в вашей системе.
     *
     * @return bool True если идентификатор покупателя был установлен, false если нет
     */
    public function hasMerchantCustomerId(): bool
    {
        return null !== $this->_merchant_customer_id;
    }

    /**
     * Устанавливает идентификатор покупателя в вашей системе.
     *
     * @param string|null $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов
     */
    public function setMerchantCustomerId(?string $merchant_customer_id): self
    {
        $this->_merchant_customer_id = $this->validatePropertyValue('_merchant_customer_id', $merchant_customer_id);
        return $this;
    }

    /**
     * Возвращает платежное поручение.
     *
     * @return AbstractPaymentOrder|null
     */
    public function getPaymentOrder(): ?AbstractPaymentOrder
    {
        return $this->_payment_order;
    }

    /**
     * Проверяет, было ли установлено платежное поручение.
     *
     * @return bool True если платежное поручение было установлены, false если нет
     */
    public function hasPaymentOrder(): bool
    {
        return null !== $this->_payment_order;
    }

    /**
     * Устанавливает платежное поручение.
     *
     * @param AbstractPaymentOrder|array|null $payment_order Платежное поручение
     *
     * @return self
     */
    public function setPaymentOrder(mixed $payment_order = null): self
    {
        if (is_array($payment_order)) {
            $payment_order = (new PaymentOrderFactory())->factoryFromArray($payment_order);
        }
        $this->_payment_order = $this->validatePropertyValue('_payment_order', $payment_order);
        return $this;
    }

    /**
     * Возвращает реквизиты получателя оплаты.
     *
     * @return AbstractReceiver|null Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона.
     */
    public function getReceiver(): ?AbstractReceiver
    {
        return $this->_receiver;
    }

    /**
     * Проверяет, были ли установлены реквизиты получателя оплаты.
     *
     * @return bool True если реквизиты получателя оплаты были установлены, false если нет
     */
    public function hasReceiver(): bool
    {
        return null !== $this->_receiver;
    }

    /**
     * Устанавливает реквизиты получателя оплаты.
     *
     * @param AbstractReceiver|array|null $receiver Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона.
     *
     * @return self
     */
    public function setReceiver(mixed $receiver = null): self
    {
        if (is_array($receiver)) {
            $factory = new ReceiverFactory();
            $receiver = $factory->factoryFromArray($receiver);
        }
        $this->_receiver = $this->validatePropertyValue('_receiver', $receiver);
        return $this;
    }

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool
    {
        if (!parent::validate()) {
            return false;
        }
        if ($this->hasPaymentToken()) {
            if ($this->hasPaymentMethodId()) {
                $this->setValidationError('Both paymentToken and paymentMethodID values are specified');

                return false;
            }
            if ($this->hasPaymentMethodData()) {
                $this->setValidationError('Both paymentToken and paymentData values are specified');

                return false;
            }
        } elseif ($this->hasPaymentMethodId()) {
            if ($this->hasPaymentMethodData()) {
                $this->setValidationError('Both paymentMethodID and paymentData values are specified');

                return false;
            }
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания платежа.
     *
     * @return CreatePaymentRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreatePaymentRequestBuilder
    {
        return new CreatePaymentRequestBuilder();
    }
}
