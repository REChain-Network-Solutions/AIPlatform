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

namespace YooKassa\Model\SavePaymentMethod;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Payment\PaymentMethod\BankCard;
use YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation;
use YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SavePaymentMethod.
 *
 * Сохраненный способ оплаты.
 * 
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property SavePaymentMethodType $type Код способа оплаты.  Возможное значение: ~`bank_card` — банковская карта.
 * @property BankCard $card Данные банковской карты
 * @property string $id Идентификатор сохраненного способа оплаты.
 * @property bool $saved Признак сохранения способа оплаты для [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
 * @property SavePaymentMethodStatus $status Статус проверки и сохранения способа оплаты.
 * @property SavePaymentMethodHolder $holder Данные магазина, для которого сохраняется способ оплаты.
 * @property string $title Название способа оплаты.
 * @property AbstractConfirmation $confirmation Выбранный сценарий подтверждения привязки. Присутствует, когда привязка ожидает подтверждения от пользователя.
*/
class SavePaymentMethod extends AbstractObject implements SavePaymentMethodInterface
{
    /**
     * Код способа оплаты.  Возможное значение: ~`bank_card` — банковская карта.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [SavePaymentMethodType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_type = null;

    /**
     * Данные банковской карты
     *
     * @var BankCard|null
     */
    #[Assert\Valid]
    #[Assert\Type(BankCard::class)]
    protected ?BankCard $_card = null;

    /**
     * Идентификатор сохраненного способа оплаты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    protected ?string $_id = null;

    /**
     * Признак сохранения способа оплаты для [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
     * * true — способ оплаты сохранен для автоплатежей и выплат;
     * * false — способ оплаты не сохранен.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected ?bool $_saved = false;

    /**
     * Статус проверки и сохранения способа оплаты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [SavePaymentMethodStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = null;

    /**
     * Данные магазина, для которого сохраняется способ оплаты.
     *
     * @var SavePaymentMethodHolder|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(SavePaymentMethodHolder::class)]
    protected ?SavePaymentMethodHolder $_holder = null;

    /**
     * Название способа оплаты.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    protected ?string $_title = null;

    /**
     * @var AbstractConfirmation|null
     */
    #[Assert\Type(AbstractConfirmation::class)]
    protected ?AbstractConfirmation $_confirmation = null;

    /**
     * Возвращает type.
     *
     * @return string|null
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает type.
     *
     * @param string|null $type Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает card.
     *
     * @return BankCard|null
     */
    public function getCard(): ?BankCard
    {
        return $this->_card;
    }

    /**
     * Устанавливает card.
     *
     * @param BankCard|array|null $card Данные банковской карты.
     *
     * @return self
     */
    public function setCard(mixed $card = null): self
    {
        $this->_card = $this->validatePropertyValue('_card', $card);
        return $this;
    }

    /**
     * Возвращает id.
     *
     * @return string|null
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор сохраненного способа оплаты.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает saved.
     *
     * @return bool|null
     */
    public function getSaved(): ?bool
    {
        return $this->_saved;
    }

    /**
     * Устанавливает saved.
     *
     * @param bool|null $saved Признак сохранения способа оплаты для [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
     *
     * @return self
     */
    public function setSaved(?bool $saved = null): self
    {
        $this->_saved = $this->validatePropertyValue('_saved', $saved);
        return $this;
    }

    /**
     * Возвращает status.
     *
     * @return string|null
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает status.
     *
     * @param string|null $status Статус проверки и сохранения способа оплаты.
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает holder.
     *
     * @return SavePaymentMethodHolder|null
     */
    public function getHolder(): ?SavePaymentMethodHolder
    {
        return $this->_holder;
    }

    /**
     * Устанавливает holder.
     *
     * @param SavePaymentMethodHolder|array|null $holder Данные магазина, для которого сохраняется способ оплаты.
     *
     * @return self
     */
    public function setHolder(mixed $holder = null): self
    {
        $this->_holder = $this->validatePropertyValue('_holder', $holder);
        return $this;
    }

    /**
     * Возвращает title.
     *
     * @return string|null
     */
    public function getTitle(): ?string
    {
        return $this->_title;
    }

    /**
     * Устанавливает title.
     *
     * @param string|null $title Название способа оплаты.
     *
     * @return self
     */
    public function setTitle(?string $title = null): self
    {
        $this->_title = $this->validatePropertyValue('_title', $title);
        return $this;
    }

    /**
     * Возвращает confirmation.
     *
     * @return AbstractConfirmation|null
     */
    public function getConfirmation(): ?AbstractConfirmation
    {
        return $this->_confirmation;
    }

    /**
     * Устанавливает confirmation.
     *
     * @param AbstractConfirmation|array|null $confirmation Выбранный сценарий подтверждения привязки.
     *
     * @return self
     */
    public function setConfirmation(mixed $confirmation = null): self
    {
        if (is_array($confirmation)) {
            $confirmation = (new ConfirmationFactory())->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

}

