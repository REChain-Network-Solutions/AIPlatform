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

namespace YooKassa\Request\Refunds;

use YooKassa\Common\AbstractRequest;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\RefundDealData;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Model\Refund\Source;
use YooKassa\Model\Refund\SourceInterface;
use YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData;
use YooKassa\Request\Refunds\RefundMethodData\RefundMethodDataFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreateRefundRequest.
 *
 * Класс объекта запроса для создания возврата.
 *
 * @example 02-builder.php 147 33 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $payment_id Идентификатор платежа для которого создаётся возврат
 * @property string $paymentId Идентификатор платежа для которого создаётся возврат
 * @property AmountInterface $amount Сумма возврата
 * @property string $description Комментарий к операции возврата, основание для возврата средств покупателю.
 * @property null|ReceiptInterface $receipt Инстанс чека или null
 * @property null|ListObjectInterface|SourceInterface[] $sources Информация о распределении денег — сколько и в какой магазин нужно перевести
 * @property null|RefundDealData $deal Информация о сделке
 * @property null|AbstractRefundMethodData $refund_method_data Метод возврата
 * @property null|AbstractRefundMethodData $refundMethodData Метод возврата
 */
class CreateRefundRequest extends AbstractRequest implements CreateRefundRequestInterface
{
    /**
     * @var string|null Идентификатор платежа для которого создаётся возврат
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 36)]
    #[Assert\Length(min: 36)]
    private ?string $_payment_id = null;

    /**
     * @var AmountInterface|null Сумма возврата
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * @var ReceiptInterface|null Чек возврата
     */
    #[Assert\Valid]
    #[Assert\Type(Receipt::class)]
    private ?ReceiptInterface $_receipt = null;

    /**
     * @var string|null Комментарий к операции возврата, основание для возврата средств покупателю
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 250)]
    private ?string $_description = null;

    /**
     * @var SourceInterface[]|ListObjectInterface|null Информация о распределении денег
     */
    #[Assert\Valid]
    #[Assert\AllType(Source::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObjectInterface $_sources = null;

    /**
     * @var RefundDealData|null Данные о сделке, в составе которой проходит возврат
     */
    #[Assert\Valid]
    #[Assert\Type(RefundDealData::class)]
    private ?RefundDealData $_deal = null;

    /**
     * @var AbstractRefundMethodData|null
     */
    #[Assert\Type(AbstractRefundMethodData::class)]
    private ?AbstractRefundMethodData $_refund_method_data = null;

    /**
     * Возвращает идентификатор платежа для которого создаётся возврат средств.
     *
     * @return string|null Идентификатор платежа для которого создаётся возврат
     */
    public function getPaymentId(): ?string
    {
        return $this->_payment_id;
    }

    /**
     * Проверяет, был ли установлена идентификатор платежа.
     *
     * @return bool True если идентификатор платежа был установлен, false если нет
     */
    public function hasPaymentId(): bool
    {
        return !empty($this->_payment_id);
    }

    /**
     * Устанавливает идентификатор платежа для которого создаётся возврат
     *
     * @param string|null $payment_id Идентификатор платежа
     *
     * @return self
     */
    public function setPaymentId(?string $payment_id = null): self
    {
        $this->_payment_id = $this->validatePropertyValue('_payment_id', $payment_id);
        return $this;
    }

    /**
     * Возвращает сумму возврата.
     *
     * @return AmountInterface|null Сумма возврата
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Проверяет, была ли установлена сумма возврата.
     *
     * @return bool True если сумма возврата была установлена, false если нет
     */
    public function hasAmount(): bool
    {
        return !empty($this->_amount);
    }

    /**
     * Устанавливает сумму.
     *
     * @param AmountInterface|array|string $amount Сумма возврата
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает чек, если он есть.
     *
     * @return null|ReceiptInterface Данные фискального чека 54-ФЗ или null, если чека нет
     */
    public function getReceipt(): ?ReceiptInterface
    {
        return $this->_receipt;
    }

    /**
     * Устанавливает чек.
     *
     * @param null|array|ReceiptInterface $receipt Инстанс чека или null для удаления информации о чеке
     *
     * @return self
     */
    public function setReceipt(mixed $receipt = null): self
    {
        $this->_receipt = $this->validatePropertyValue('_receipt', $receipt);
        return $this;
    }

    /**
     * Проверяет наличие чека.
     *
     * @return bool True если чек есть, false если нет
     */
    public function hasReceipt(): bool
    {
        return null !== $this->_receipt && $this->_receipt->notEmpty();
    }

    /**
     * Удаляет чек из запроса.
     */
    public function removeReceipt(): void
    {
        $this->setReceipt(null);
    }

    /**
     * Возвращает комментарий к возврату или null, если комментарий не задан.
     *
     * @return string|null Комментарий к операции возврата, основание для возврата средств покупателю
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Проверяет задан ли комментарий к создаваемому возврату.
     *
     * @return bool True если комментарий установлен, false если нет
     */
    public function hasDescription(): bool
    {
        return null !== $this->_description;
    }

    /**
     * Устанавливает комментарий к возврату.
     *
     * @param string|null $description Комментарий к операции возврата, основание для возврата средств покупателю
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Устанавливает sources (массив распределения денег между магазинами).
     *
     * @param array|ListObjectInterface|null $sources Массив распределения денег между магазинами
     *
     * @return self
     */
    public function setSources(mixed $sources = null): self
    {
        $this->_sources = $this->validatePropertyValue('_sources', $sources);
        return $this;
    }

    /**
     * Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.
     *
     * @return SourceInterface[]|ListObjectInterface Информация о распределении денег
     */
    public function getSources(): ListObjectInterface
    {
        if ($this->_sources === null) {
            $this->_sources = new ListObject(Source::class);
        }
        return $this->_sources;
    }

    /**
     * Проверяет наличие информации о распределении денег.
     */
    public function hasSources(): bool
    {
        return !empty($this->_sources);
    }

    /**
     * Возвращает данные о сделке, в составе которой проходит возврат
     *
     * @return RefundDealData|null Данные о сделке, в составе которой проходит возврат
     */
    public function getDeal(): ?RefundDealData
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
     * Устанавливает данные о сделке, в составе которой проходит возврат
     *
     * @param null|array|RefundDealData $deal Данные о сделке, в составе которой проходит возврат
     *
     * @return self
     */
    public function setDeal(mixed $deal = null): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает метод возврата.
     *
     * @return AbstractRefundMethodData|null Метод возврата
     */
    public function getRefundMethodData(): ?AbstractRefundMethodData
    {
        return $this->_refund_method_data;
    }

    /**
     * Проверяет установлен ли объект с методом возврата.
     *
     * @return bool True если объект метода возврата установлен, false если нет
     */
    public function hasRefundMethodData(): bool
    {
        return !empty($this->_refund_method_data);
    }

    /**
     * Устанавливает метод возврата.
     *
     * @param AbstractRefundMethodData|array|null $refund_method_data Метод возврата
     *
     * @return self
     */
    public function setRefundMethodData(mixed $refund_method_data = null): self
    {
        if (is_array($refund_method_data)) {
            $refund_method_data = (new RefundMethodDataFactory)->factoryFromArray($refund_method_data);
        }
        $this->_refund_method_data = $this->validatePropertyValue('_refund_method_data', $refund_method_data);
        return $this;
    }

    /**
     * Валидирует объект запроса.
     *
     * @return bool True если запрос валиден и его можно отправить в API, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasPaymentId()) {
            $this->setValidationError('Payment id not specified');

            return false;
        }

        if (!$this->hasAmount()) {
            $this->setValidationError('Refund amount not specified');

            return false;
        }

        $value = $this->_amount->getValue();
        if (empty($value) || $value <= 0.0) {
            $this->setValidationError('Invalid refund amount value: ' . $value);

            return false;
        }

        if ($this->hasReceipt() && $this->getReceipt()?->notEmpty()) {
            $email = $this->getReceipt()?->getCustomer()?->getEmail();
            $phone = $this->getReceipt()?->getCustomer()?->getPhone();
            if (empty($email) && empty($phone)) {
                $this->setValidationError('Both email and phone values are empty in receipt');

                return false;
            }
        }

        return true;
    }

    /**
     * Возвращает билдер объектов текущего типа.
     *
     * @return CreateRefundRequestBuilder Инстанс билдера запросов
     */
    public static function builder(): CreateRefundRequestBuilder
    {
        return new CreateRefundRequestBuilder();
    }
}
