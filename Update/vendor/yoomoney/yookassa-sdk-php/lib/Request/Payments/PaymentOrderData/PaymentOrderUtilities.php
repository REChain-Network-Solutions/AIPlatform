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

namespace YooKassa\Request\Payments\PaymentOrderData;

use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Request\Payments\PaymentPeriod;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentOrderUtilities.
 *
 * Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.
 *
 * Необходимо передавать при [оплате ЖКУ](/developers/payment-acceptance/scenario-extensions/utility-payments).
 *
 * Кроме параметров, отмеченных как обязательные, должен быть передан как минимум один параметр из этого списка: `payment_document_id`, `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $type Код вида платежного поручения.
 * @property MonetaryAmount $amount Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа. Равна общей сумме платежа.
 * @property string $payment_purpose Назначение платежа (не больше 210 символов).
 * @property PaymentOrderRecipient $recipient Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг.
 * @property string $kbk Код бюджетной классификации (КБК).
 * @property string $oktmo Код ОКТМО (Общероссийский классификатор территорий муниципальных образований).
 * @property PaymentPeriod $payment_period Период оплаты, за который выставлены начисления и за который вносится оплата.
 * @property string $payment_document_id Идентификатор платежного документа.  Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.
 * @property string $payment_document_number Номер платежного документа на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`.
 * @property string $account_number Номер лицевого счета на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`.
 * @property string $unified_account_number Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение».  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`.
 * @property string $service_id Идентификатор жилищно-коммунальной услуги (ЖКУ).  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`.
 */
class PaymentOrderUtilities extends AbstractPaymentOrder
{
    /**
     * Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа.
     * Равна общей сумме платежа.
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Назначение платежа (не больше 210 символов).
     * Рекомендуется формировать с учетом рекомендаций из [Письма Банка России № ИН-04-45|12 от 22.02.2018](https://my.dom.gosuslugi.ru/filestore/publicDownloadServlet?context=contentmanagement&uid=ef9a477a-2beb-4212-be30-aed231160db1&mode=view).
     * Пример: ~`Оплата ЖКХ;ЕЛС 80KX478547;ПРД 12.2024;Иванов Иван;г.Москва, ул.Флотская, д.1, кв.1`
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_payment_purpose = null;

    /**
     * Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг.
     *
     * @var PaymentOrderRecipient|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(PaymentOrderRecipient::class)]
    private ?PaymentOrderRecipient $_recipient = null;

    /**
     * Код бюджетной классификации (КБК).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 20)]
    #[Assert\Length(min: 1)]
    #[Assert\Regex("/^(\d{20})|(0)$/")]
    private ?string $_kbk = null;

    /**
     * Код ОКТМО (Общероссийский классификатор территорий муниципальных образований).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 8)]
    #[Assert\Length(min: 1)]
    #[Assert\Regex("/^(\d{8})|(0)$/")]
    private ?string $_oktmo = null;

    /**
     * Период оплаты, за который выставлены начисления и за который вносится оплата.
     *
     * @var PaymentPeriod|null
     */
    #[Assert\Type(PaymentPeriod::class)]
    private ?PaymentPeriod $_payment_period = null;

    /**
     * Идентификатор платежного документа.
     * Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 18)]
    #[Assert\Length(min: 18)]
    private ?string $_payment_document_id = null;

    /**
     * Номер платежного документа на стороне поставщика ЖКУ.
     * Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 30)]
    #[Assert\Length(min: 1)]
    #[Assert\Regex("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u")]
    private ?string $_payment_document_number = null;

    /**
     * Номер лицевого счета на стороне поставщика ЖКУ.
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 30)]
    #[Assert\Length(min: 1)]
    #[Assert\Regex("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u")]
    private ?string $_account_number = null;

    /**
     * Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение».
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 10)]
    #[Assert\Length(min: 10)]
    private ?string $_unified_account_number = null;

    /**
     * Идентификатор жилищно-коммунальной услуги (ЖКУ).
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 13)]
    #[Assert\Length(min: 13)]
    private ?string $_service_id = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentOrderType::UTILITIES);
    }

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает amount.
     *
     * @param AmountInterface|array|null $amount Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа.
     * Равна общей сумме платежа.
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает payment_purpose.
     *
     * @return string|null
     */
    public function getPaymentPurpose(): ?string
    {
        return $this->_payment_purpose;
    }

    /**
     * Устанавливает payment_purpose.
     *
     * @param string|array|null $payment_purpose Назначение платежа (не больше 210 символов).
     *
     * @return self
     */
    public function setPaymentPurpose(?string $payment_purpose = null): self
    {
        $this->_payment_purpose = $this->validatePropertyValue('_payment_purpose', $payment_purpose);
        return $this;
    }

    /**
     * Возвращает recipient.
     *
     * @return PaymentOrderRecipient|null
     */
    public function getRecipient(): ?PaymentOrderRecipient
    {
        return $this->_recipient;
    }

    /**
     * Устанавливает recipient.
     *
     * @param PaymentOrderRecipient|array|null $recipient Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг.
     *
     * @return self
     */
    public function setRecipient(mixed $recipient = null): self
    {
        $this->_recipient = $this->validatePropertyValue('_recipient', $recipient);
        return $this;
    }

    /**
     * Возвращает kbk.
     *
     * @return string|null
     */
    public function getKbk(): ?string
    {
        return $this->_kbk;
    }

    /**
     * Устанавливает kbk.
     *
     * @param string|null $kbk Код бюджетной классификации (КБК).
     *
     * @return self
     */
    public function setKbk(?string $kbk = null): self
    {
        $this->_kbk = $this->validatePropertyValue('_kbk', $kbk);
        return $this;
    }

    /**
     * Возвращает oktmo.
     *
     * @return string|null
     */
    public function getOktmo(): ?string
    {
        return $this->_oktmo;
    }

    /**
     * Устанавливает oktmo.
     *
     * @param string|null $oktmo Код ОКТМО (Общероссийский классификатор территорий муниципальных образований).
     *
     * @return self
     */
    public function setOktmo(?string $oktmo = null): self
    {
        $this->_oktmo = $this->validatePropertyValue('_oktmo', $oktmo);
        return $this;
    }

    /**
     * Возвращает payment_period.
     *
     * @return PaymentPeriod|null
     */
    public function getPaymentPeriod(): ?PaymentPeriod
    {
        return $this->_payment_period;
    }

    /**
     * Устанавливает payment_period.
     *
     * @param PaymentPeriod|array|null $payment_period Период оплаты, за который выставлены начисления и за который вносится оплата.
     *
     * @return self
     */
    public function setPaymentPeriod(mixed $payment_period = null): self
    {
        $this->_payment_period = $this->validatePropertyValue('_payment_period', $payment_period);
        return $this;
    }

    /**
     * Возвращает payment_document_id.
     *
     * @return string|null
     */
    public function getPaymentDocumentId(): ?string
    {
        return $this->_payment_document_id;
    }

    /**
     * Устанавливает идентификатор платежного документа.
     *
     * @param string|null $payment_document_id Идентификатор платежного документа.
     * Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.
     *
     * @return self
     */
    public function setPaymentDocumentId(?string $payment_document_id = null): self
    {
        $this->_payment_document_id = $this->validatePropertyValue('_payment_document_id', $payment_document_id);
        return $this;
    }

    /**
     * Проверяет, были ли установлен идентификатор платежного документа.
     *
     * @return bool True если идентификатор платежного документа был установлен, false если нет
     */
    public function hasPaymentDocumentId(): bool
    {
        return null !== $this->_payment_document_id;
    }

    /**
     * Возвращает payment_document_number.
     *
     * @return string|null
     */
    public function getPaymentDocumentNumber(): ?string
    {
        return $this->_payment_document_number;
    }

    /**
     * Устанавливает номер платежного документа на стороне поставщика ЖКУ.
     *
     * @param string|null $payment_document_number Номер платежного документа на стороне поставщика ЖКУ.
     * Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`.
     *
     * @return self
     */
    public function setPaymentDocumentNumber(?string $payment_document_number = null): self
    {
        $this->_payment_document_number = $this->validatePropertyValue('_payment_document_number', $payment_document_number);
        return $this;
    }

    /**
     * Проверяет, были ли установлен номер платежного документа на стороне поставщика ЖКУ.
     *
     * @return bool True если номер платежного документа на стороне поставщика ЖКУ был установлен, false если нет
     */
    public function hasPaymentDocumentNumber(): bool
    {
        return null !== $this->_payment_document_number;
    }

    /**
     * Возвращает номер лицевого счета на стороне поставщика ЖКУ.
     *
     * @return string|null
     */
    public function getAccountNumber(): ?string
    {
        return $this->_account_number;
    }

    /**
     * Устанавливает номер лицевого счета на стороне поставщика ЖКУ.
     *
     * @param string|null $account_number Номер лицевого счета на стороне поставщика ЖКУ.
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`.
     *
     * @return self
     */
    public function setAccountNumber(?string $account_number = null): self
    {
        $this->_account_number = $this->validatePropertyValue('_account_number', $account_number);
        return $this;
    }

    /**
     * Проверяет, были ли установлен номер лицевого счета на стороне поставщика ЖКУ.
     *
     * @return bool True если номер лицевого счета на стороне поставщика ЖКУ был установлен, false если нет
     */
    public function hasAccountNumber(): bool
    {
        return null !== $this->_account_number;
    }

    /**
     * Возвращает единый лицевой счет.
     *
     * @return string|null
     */
    public function getUnifiedAccountNumber(): ?string
    {
        return $this->_unified_account_number;
    }

    /**
     * Устанавливает единый лицевой счет.
     *
     * @param string|null $unified_account_number Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение».
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`.
     *
     * @return self
     */
    public function setUnifiedAccountNumber(?string $unified_account_number = null): self
    {
        $this->_unified_account_number = $this->validatePropertyValue('_unified_account_number', $unified_account_number);
        return $this;
    }

    /**
     * Проверяет, были ли установлен единый лицевой счет.
     *
     * @return bool True если единый лицевой счет был установлен, false если нет
     */
    public function hasUnifiedAccountNumber(): bool
    {
        return null !== $this->_unified_account_number;
    }

    /**
     * Возвращает идентификатор жилищно-коммунальной услуги (ЖКУ).
     *
     * @return string|null
     */
    public function getServiceId(): ?string
    {
        return $this->_service_id;
    }

    /**
     * Устанавливает идентификатор жилищно-коммунальной услуги (ЖКУ).
     *
     * @param string|null $service_id Идентификатор жилищно-коммунальной услуги (ЖКУ).
     * Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`.
     *
     * @return self
     */
    public function setServiceId(?string $service_id = null): self
    {
        $this->_service_id = $this->validatePropertyValue('_service_id', $service_id);
        return $this;
    }

    /**
     * Проверяет, были ли установлен идентификатор жилищно-коммунальной услуги (ЖКУ).
     *
     * @return bool True если идентификатор жилищно-коммунальной услуги (ЖКУ) был установлен, false если нет
     */
    public function hasServiceId(): bool
    {
        return null !== $this->service_id;
    }

    /**
     * Проверяет, что все необходимые поля заданы.
     *
     * @return bool
     */
    public function validate(): bool
    {
        return $this->hasPaymentDocumentId()
            || $this->hasPaymentDocumentNumber()
            || $this->hasAccountNumber()
            || $this->hasUnifiedAccountNumber()
            || $this->hasServiceId();
    }
}
