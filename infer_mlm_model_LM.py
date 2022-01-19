import argparse
import random
import torch
import torch.optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import tokenizer
import html
import util.cfg
import util.seed
import util.dist
from model import MODEL_OPT
from dataset import DSET_OPT
from tokenizer import TKNZR_OPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=int)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--p_mask', required=False, default=0.15, type=float)
    parser.add_argument('--p_len', required=False, default=0.1, type=float)
    parser.add_argument('--max_span_len', required=False, default=9, type=int)
    # parser.add_argument('--src', required=True, action='append')
    args = parser.parse_args()

    util.seed.set_seed(seed=args.seed)

    exp_cfg = util.cfg.load(exp_name=args.exp_name)
    print(exp_cfg)
    # Load dataset and dataset config.
    # dset = DSET_OPT[exp_cfg.dataset_type](
    #     exp_cfg.dataset_exp_name, 10000
    # )

    dset = [
        ('<loc0>鳳林警協助發放振興五倍券<per0>頒加菜金','<org0>轄區內<num>處偏鄉部落因附近沒有郵局及便利商店,由當地派出所提供逾<num>人預約、領券服務,<org1><per0>今天前往慰勉員警,頒發加菜金。因應<en>-<num>造成百業蕭條,政府為刺激消費帶動景氣,推出振興五倍券,提供線上預約、便利商店、郵局預約領券等多元管道,但仍有地區因人口老化等因素,由員警協助預約發放。<org2>轄內的<loc1>、<loc2>、<loc3>、<loc4>及<loc5>等<num>村落地處偏遠,分別由西林派出所、萬榮分駐所及紅葉派出所員警協助辦理紙本五倍券預約及發放。<per0>今天前往萬榮分駐所,慰勉基層員警,感謝員警協助推動五倍券發放,發揮警察服務精神,協助偏鄉部落民眾,也頒發加菜金給<org2><num>個協助發放工作的分駐所,鼓舞工作士氣。鳳林分局長<per1>表示,<num>處派出所、分駐所結合村里辦公處加強宣傳,配合防疫政策妥適安排動線及民眾等待區,共協助第一階段<num>多名居民預約紙本五倍券,接下來將繼續協助領券工作。'),
        ('總統府光雕秀看這裡重現日文「<loc0>」、<loc1>動物登場', '<num>年雙十國慶將至,總統<per0><num>日晚間將親自出席<org0>主辦的總統府建築光雕展演。據悉,今年展演文總安排諸多巧思與「彩蛋」,除了奧運、帕運<loc0>英雄的畫面,日媒主播錄製「<loc0>」,同時也為感謝國際友人在疫情期間伸出援手,象徵<num>國的可愛動物<en>也會登場。《<unk>》安排全程直播。<org0>規劃總統府建築光雕展演於<num>日晚間<num>:<num>正式點燈,以「百年追求、世界<loc0>」為主軸,將展現四大主題理念,包括「自覺<loc0>、自律<loc0>、自信<loc0>、世界<loc0>」,向<loc0>文化前輩致敬。現場的工作人員表示,今年<loc0>經歷許多重要轉折,包括疫情擴散、在國際盟友的互助合作下穩住陣腳,我國的奧運及<org1>又在<loc2>大放異彩,加上今年適逢<org2>成立百年紀念的歷史意義,讓擔任今年光雕的策展人的<org3>副執行長<per1>,都直呼,「今年<loc0>的哏多到爆棚!」據了解,今年文總安排的巧思與彩蛋確實相當多,在光雕展演的前段,先以文協百年的主題揭開序幕,中段則是重現民眾幾個月前面臨防疫三級警戒的日常生活,除了空蕩蕩的街道、視訊上班上課、量體溫、酒精消毒、掃實聯制<en>等,還有外送員穿梭街頭送餐的動畫。'),
        ('<loc0>上演真實版《<unk>》!民眾體驗打起來警鎮壓竄逃畫面曝', '<loc1>劇集《<unk>》全球爆紅,繼梨泰院地鐵站增設戲劇體驗的設施供劇迷朝聖後,<loc2>所拍攝的預告片也利用劇中「<num>木頭人娃娃」的恐懼感,禁止人民在過紅綠燈時穿越馬路,創新的宣傳手法吸引目光。近來,<loc0>也在當地建了一間遊戲體驗的快閃咖啡廳,<num>日、<num>日於<loc3>登場,怎料引起暴動,場面混亂、警察出動鎮壓,畫面曝光後,令不少網友驚呼:「他們真的體驗到《<unk>》了!」<loc0><en>片商為宣傳《<unk>》,實際於<loc3>設置出一間體驗咖啡廳,據悉,館內高度還原戲劇裡出現的大型豬存錢筒、劇裡服裝以及戳椪糖關卡,馬上就吸引大批劇迷朝聖,店外車水馬龍。然而,根據當地媒體報導,該咖啡廳人氣太旺,若想進入館內得從前一晚就露宿排隊,正因如此,現場出現了「插隊」動亂,使得一群人在館外打起來,而民眾為了躲避打架紛爭、邊尖叫邊逃難,火爆程度引來警方鎮壓,各種動亂畫面在推特、抖音上流傳,最終結局就是該體驗館提早關閉。'),
        ('囤房大戶<num>人<org0>加強查緝租金所得漏報', '配合<org1><org2>臨時提案,<org0>宣布即日啟動「個人間房屋租賃所得專案查核作業計畫」,首度精準鎖定全國擁有<num>戶以上非自住房屋的<num>名囤房大戶,展開租金所得查核,遏止漏稅。<org0>今天發布新聞稿,並同步於<org3>例行記者會上說明,配合<org1><org4>今年<num>月<num>日通過的臨時提案,將鎖定全國持有<num>戶以上非自住房屋者,逐步清查是否如實申報租金收入,以落實居住正義,避免租金收入成為逃漏稅黑洞。<org3>署長<per0>表示,統計目前全<loc0>持有逾<num>戶以上非自住房屋的個人共<num>名,五區<org5>近日將陸續逐一寄發輔導函,要求前來說明名下房屋是否有出租情形若有出租但未曾就租賃所得繳稅,將輔導補報補繳,此階段僅需補稅仍免罰。'),
        ('用飲水機洗臉的貓?短腿大眼萌樣紅到國外「原來是美容祕訣!」', '大家對貓咪的印象多數是不喜歡弄濕或怕水,<loc0>卻有一隻貓咪「<en>」直接側臉躺在循環飲水機的水裡,一臉舒服的萌樣在推特上爆紅,各國的網友除了大讚<en>太可愛,也疑惑:「都弄濕了沒關係?」飼主則在貼文說<en>是「直接洗臉」的路線,有人就笑稱:「原來是美容祕訣!」現在許多飼主家中會放自動循環流水的飲水機給寵物喝水,<loc0>有位飼主<en>日前在他推特,上傳家中的曼赤肯貓「<en>」使用飲水機的影片,只見牠突然倒下躺著,頭部靠在飲水機邊,側臉臉頰直接躺在流動的水面,還伸出一隻小短腿撥弄流水,後來轉頭換下巴靠在水面,怡然自得地睜著萌萌大眼看向鏡頭。影片上傳推特後爆紅,超過<num>萬次觀看與<num>萬則轉推,不只<loc0>連各國網友都被圈粉留言,「毛都弄濕了欸…」「<en>!」「臉頰不會冷冷的嗎?」「我家也有這台,原來是這樣用的啊。」「貓咪果然是水做的。」而原<en>在貼文上則開玩笑說<en>是「直接洗臉的路線」,因此很多人讚嘆「原來這就是<en>美麗的祕密!」'),
    ]
    # dset_cfg = util.cfg.load(exp_name=exp_cfg.dataset_exp_name)

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=exp_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr](exp_name=tknzr_cfg.exp_name)
    sp_tkids = [tknzr.cls_tkid, tknzr.sep_tkid, tknzr.pad_tkid]

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    # batch_mask_tkids = []
    # batch_target_tkids = []
    # batch_is_mask = []
    # for i in [1, 1001, 2001, 3001, 4001]:
    #     (mask_tkids, target_tkids, is_mask) = dset[i]
    #     batch_mask_tkids.append(mask_tkids)
    #     batch_target_tkids.append(target_tkids)
    #     batch_is_mask.append(is_mask)
    def collate_fn(batch):
        batch_mask_tkids = []
        batch_target_tkids = []
        batch_is_mask = []
        for title, article in batch:
            target_tkids = tknzr.enc(
                txt=title,
                txt_pair=article,
                max_seq_len=exp_cfg.max_seq_len
            )

            mask_tkids = []
            is_mask = []
            mask_span_count = 0
            while sum(is_mask) == 0:
                mask_tkids = []
                is_mask = []
                mask_span_count = 0
                for tkid in target_tkids:
                    mask_span_count -= 1

                    # Masking no more than args.p_mask x 100% tokens.
                    if sum(is_mask) / len(target_tkids) >= args.p_mask:
                        mask_tkids.append(tkid)
                        is_mask.append(0)
                        continue

                    # Skip masking if current token is special token.
                    if tkid in sp_tkids:
                        mask_tkids.append(tkid)
                        is_mask.append(0)
                        continue


                    # Mask current token based on masking distribution.
                    if util.dist.mask(p=args.p_mask):
                        # Record how many tokens to be mask (span masking).
                        mask_span_count = util.dist.length(
                            p=args.p_len,
                            max_span_len=args.max_span_len
                        )
                        mask_tkids.append(tknzr.mask_tkid)
                        is_mask.append(1)
                        continue

                    # Skip masking current token.
                    mask_tkids.append(tkid)
                    is_mask.append(0)
            batch_is_mask.append(is_mask)
            batch_mask_tkids.append(mask_tkids)
            batch_target_tkids.append(target_tkids)

        return batch_mask_tkids, batch_target_tkids, batch_is_mask

    # Create data loader.
    print("initialize dataloader....")
    dldr = torch.utils.data.DataLoader(
        dset,
        batch_size=5,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=len(os.sched_getaffinity(0)),
    )

    model, model_state = MODEL_OPT[exp_cfg.model].load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **exp_cfg.__dict__,
    )
    model.eval()

    model = model.to(device)

    # B x S
    batch_mask_tkids, batch_target_tkids, batch_is_mask = next(iter(dldr))
    batch_mask_tkids = torch.LongTensor(batch_mask_tkids).to(device)
    batch_target_tkids = torch.LongTensor(batch_target_tkids).to(device)
    batch_is_mask = torch.BoolTensor(batch_is_mask).to(device)

    mask_ids = [x.nonzero(as_tuple=True)[0].tolist() for x in batch_is_mask]
    # get the max number of mask on sequences
    max_mask = max([len(x) for x in mask_ids])
    batch_out_tks = batch_mask_tkids.detach()
    for i in range(max_mask):
        # create a tensor that decide what token need to be filled
        fill_ids = torch.zeros_like(batch_mask_tkids).type(
            torch.BoolTensor).to(device)
        for B, x in enumerate(mask_ids):
            if len(x) > i:
                fill_ids[B][x[i]] = True

        # In: B, S
        # Out: B, S, V
        batch_out_probs = model.pred(batch_out_tks)

        # In: B, S, V
        # Out: B, S, K
        (
            batch_topk_tkid_probs,
            batch_topk_tkid,
        ) = batch_out_probs.topk(
            k=args.k,
            dim=-1,
        )

        # In: B, S, K
        # Out: B, S, 1
        batch_pred_tkid_cand_idx = torch.stack(
            [torch.multinomial(x, num_samples=1)
             for x in batch_topk_tkid_probs]
        )

        # In: B, S, 1
        # Out: B, S, 1
        batch_pred_tkid = torch.gather(
            batch_topk_tkid,
            -1,
            batch_pred_tkid_cand_idx
        )

        batch_out_tks = torch.where(
            fill_ids,
            batch_pred_tkid.squeeze(),
            batch_out_tks
        )
    # for x, y, z in zip(batch_mask_tkids[3], batch_target_tkids[3], batch_out_tks[3]):
    #     print(f'{x}\t{y}\t{z}')
    # exit()
    batch_out_tks = [tknzr.ids_to_tokens(x) for x in batch_out_tks.tolist()]
    batch_mask_tks = [tknzr.ids_to_tokens(x) for x in batch_mask_tkids.tolist()]
    batch_target_tks = [tknzr.ids_to_tokens(x) for x in batch_target_tkids.tolist()]
    mask_count = 0
    mask_acc = 0
    # for x, y, z in zip(batch_mask_tks[0], batch_target_tks[0], batch_out_tks[0]):
    #     print(f'{x}\t{y}\t{z}')
    # exit()
    generations = []
    print('<table>')
    for text, target, pred in zip(batch_mask_tks, batch_target_tks, batch_out_tks):
        sentence = ''
        print('<tr><th>text</th><th>target</th><th>pred</th></tr>')
        for a, b, c in zip(text, target, pred,):
            sentence += c
            if a == "<mask>":
                mask_count += 1
                sentence += '=' + c + '='
                if b == c:
                    mask_acc += 1
                else:
                    c += "<ERROR>"

            print('<tr>')
            print(
                f'<td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td>')
            print('</tr>')
            generations.append(sentence.replace('==', ''))

    print('</table>')
    
    for text in gnerations:
        print(text)


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python infer_mlm_model_LM.py \
    --ckpt 200000 \
    --exp_name mlm_2M_l12 \
    --k 1 \
    --seed 42 >sent_LM.html
"""



"""
<loc0>鳳林警協助發放振興五倍券<per0>頒加菜金
<org0>轄區內<num>處偏鄉部落因附近沒有郵局及便利商店,<|split|>由當地派出所提供逾<num>人預約、領券服務<|split|>,<org1><per0>今天前往慰勉員警,頒發加菜金。因應<en>-<num>造成百業蕭條,<|split|>政府為刺激消費帶動景氣<|split|>,推出振興五倍券,提供線上預約、便利商店、郵局預約領券等多元管道,但仍有地區因人口老化等因素,<|split|>由員警協助預約發放<|split|>。<org2>轄內的<loc1>、<loc2>、<loc3>、<loc4>及<loc5>等<num>村落地處偏遠,<|split|>分別由西林派出所、萬榮分駐所及紅葉派出所員警協助辦理紙本五倍券預約及發放<|split|>。<per0>今天前往萬榮分駐所,慰勉基層員警,感謝員警協助推動五倍券發放,<|split|>發揮警察服務精神<|split|>,協助偏鄉部落民眾,也頒發加菜金給<org2><num>個協助發放工作的分駐所,鼓舞工作士氣。鳳林分局長<per1>表示,<num>處派出所、分駐所結合村里辦公處加強宣傳,配合防疫政策妥適安排動線及民眾等待區,共協助第一階段<num>多名居民預約紙本五倍券,接下來將繼續協助領券工作。

總統府光雕秀看這裡重現日文「<loc0>」、<loc1>動物登場
<num>年雙十國慶將至,總統<per0><num>日晚間將親自出席<org0>主辦的總統府建築光雕展演。據悉,<|split|>今年展演文總安排諸多巧思與「彩蛋」<|split|>,除了奧運、帕運<loc0>英雄的畫面,日媒主播錄製「<loc0>」,<|split|>同時也為感謝國際友人在疫情期間伸出援手<|split|>,象徵<num>國的可愛動物<en>也會登場。《<unk>》安排全程直播。<|split|><org0>規劃總統府建築光雕展演於<num>日晚間<num>:<num>正式點燈<|split|>,以「百年追求、世界<loc0>」為主軸,<|split|>將展現四大主題理念<|split|>,包括「自覺<loc0>、自律<loc0>、自信<loc0>、世界<loc0>」,向<loc0>文化前輩致敬。現場的工作人員表示,今年<loc0>經歷許多重要轉折,包括疫情擴散、在國際盟友的互助合作下穩住陣腳,我國的奧運及<org1>又在<loc2>大放異彩,<|split|>加上今年適逢<org2>成立百年紀念的歷史意義<|split|>,讓擔任今年光雕的策展人的<org3>副執行長<per1>,都直呼,「今年<loc0>的哏多到爆棚!」據了解,<|split|>今年文總安排的巧思與彩蛋確實相當多<|split|>,在光雕展演的前段,先以文協百年的主題揭開序幕,<|split|>中段則是重現民眾幾個月前面臨防疫三級警戒的日常生活<|split|>,除了空蕩蕩的街道、視訊上班上課、量體溫、酒精消毒、掃實聯制<en>等,還有外送員穿梭街頭送餐的動畫。

<loc0>上演真實版《<unk>》!民眾體驗打起來警鎮壓竄逃畫面曝
<loc1>劇集《<unk>》全球爆紅,繼梨泰院地鐵站增設戲劇體驗的設施供劇迷朝聖後,<loc2>所拍攝的預告片也利用劇中「<num>木頭人娃娃」的恐懼感,禁止人民在過紅綠燈時穿越馬路,<|split|>創新的宣傳手法吸引目光<|split|>。近來,<loc0>也在當地建了一間遊戲體驗的快閃咖啡廳,<num>日、<num>日於<loc3>登場,<|split|>怎料引起暴動<|split|>,場面混亂、警察出動鎮壓,<|split|>畫面曝光後<|split|>,令不少網友驚呼:「<|split|>他們真的體驗到《<unk>》了!<|split|>」<loc0><en>片商為宣傳《<unk>》,<|split|>實際於<loc3>設置出一間體驗咖啡廳<|split|>,據悉,<|split|>館內高度還原戲劇裡出現的大型豬存錢筒、劇裡服裝以及戳椪糖關卡<|split|>,馬上就吸引大批劇迷朝聖,店外車水馬龍。然而,根據當地媒體報導,該咖啡廳人氣太旺,若想進入館內得從前一晚就露宿排隊,正因如此,現場出現了「插隊」動亂,使得一群人在館外打起來,而民眾為了躲避打架紛爭、邊尖叫邊逃難,<|split|>火爆程度引來警方鎮壓<|split|>,各種動亂畫面在推特、抖音上流傳,最終結局就是該體驗館提早關閉。

囤房大戶<num>人<org0>加強查緝租金所得漏報
配合<org1><org2>臨時提案,<org0>宣布即日啟動「個人間房屋租賃所得專案查核作業計畫」,<|split|>首度精準鎖定全國擁有<num>戶以上非自住房屋的<num>名囤房大戶<|split|>,展開租金所得查核,遏止漏稅。<org0>今天發布新聞稿,並同步於<org3>例行記者會上說明,配合<org1><org4>今年<num>月<num>日通過的臨時提案,<|split|>將鎖定全國持有<num>戶以上非自住房屋者<|split|>,逐步清查是否如實申報租金收入,<|split|>以落實居住正義<|split|>,避免租金收入成為逃漏稅黑洞。<org3>署長<per0>表示,<|split|>統計目前全<loc0>持有逾<num>戶以上非自住房屋的個人共<num>名<|split|>,五區<org5>近日將陸續逐一寄發輔導函,<|split|>要求前來說明名下房屋是否有出租情形若有出租但未曾就租賃所得繳稅<|split|>,將輔導補報補繳,此階段僅需補稅仍免罰。

用飲水機洗臉的貓?短腿大眼萌樣紅到國外「原來是美容祕訣!」
<|split|>大家對貓咪的印象多數是不喜歡弄濕或怕水<|split|>,<loc0>卻有一隻貓咪「<en>」直接側臉躺在循環飲水機的水裡,一臉舒服的萌樣在推特上爆紅,各國的網友除了大讚<en>太可愛,也疑惑:「都弄濕了沒關係?」飼主則在貼文說<en>是「直接洗臉」的路線,有人就笑稱:「原來是美容祕訣!」現在許多飼主家中會放自動循環流水的飲水機給寵物喝水,<|split|><loc0>有位飼主<en>日前在他推特<|split|>,上傳家中的曼赤肯貓「<en>」使用飲水機的影片,<|split|>只見牠突然倒下躺著<|split|>,頭部靠在飲水機邊,側臉臉頰直接躺在流動的水面,<|split|>還伸出一隻小短腿撥弄流水,後來轉頭換下巴靠在水面,怡然自得地睜著萌萌大眼看向鏡頭<|split|>。影片上傳推特後爆紅,超過<num>萬次觀看與<num>萬則轉推,<|split|>不只<loc0>連各國網友都被圈粉留言<|split|>,「毛都弄濕了欸…」「<en>!」「臉頰不會冷冷的嗎?」「我家也有這台,原來是這樣用的啊。」「貓咪果然是水做的。」而原<en>在貼文上則開玩笑說<en>是「直接洗臉的路線」,因此很多人讚嘆「原來這就是<en>美麗的祕密!」
"""

"""
<loc0>鳳林警協助發放振興五倍券<per0>頒加菜金
<org0>轄區內<num>處偏鄉部落因附近沒有郵局及便利商店,由當地派出所提供逾<num>人預約、領券服務,<org1><per0>今天前往慰勉員警,頒發加菜金。因應<en>-<num>造成百業蕭條,政府為刺激消費帶動景氣,推出振興五倍券,提供線上預約、便利商店、郵局預約領券等多元管道,但仍有地區因人口老化等因素,由員警協助預約發放。<org2>轄內的<loc1>、<loc2>、<loc3>、<loc4>及<loc5>等<num>村落地處偏遠,分別由西林派出所、萬榮分駐所及紅葉派出所員警協助辦理紙本五倍券預約及發放。<per0>今天前往萬榮分駐所,慰勉基層員警,感謝員警協助推動五倍券發放,發揮警察服務精神,協助偏鄉部落民眾,也頒發加菜金給<org2><num>個協助發放工作的分駐所,鼓舞工作士氣。鳳林分局長<per1>表示,<num>處派出所、分駐所結合村里辦公處加強宣傳,配合防疫政策妥適安排動線及民眾等待區,共協助第一階段<num>多名居民預約紙本五倍券,接下來將繼續協助領券工作。

總統府光雕秀看這裡重現日文「<loc0>」、<loc1>動物登場
<num>年雙十國慶將至,總統<per0><num>日晚間將親自出席<org0>主辦的總統府建築光雕展演。據悉,今年展演文總安排諸多巧思與「彩蛋」,除了奧運、帕運<loc0>英雄的畫面,日媒主播錄製「<loc0>」,同時也為感謝國際友人在疫情期間伸出援手,象徵<num>國的可愛動物<en>也會登場。《<unk>》安排全程直播。<org0>規劃總統府建築光雕展演於<num>日晚間<num>:<num>正式點燈,以「百年追求、世界<loc0>」為主軸,將展現四大主題理念,包括「自覺<loc0>、自律<loc0>、自信<loc0>、世界<loc0>」,向<loc0>文化前輩致敬。現場的工作人員表示,今年<loc0>經歷許多重要轉折,包括疫情擴散、在國際盟友的互助合作下穩住陣腳,我國的奧運及<org1>又在<loc2>大放異彩,加上今年適逢<org2>成立百年紀念的歷史意義,讓擔任今年光雕的策展人的<org3>副執行長<per1>,都直呼,「今年<loc0>的哏多到爆棚!」據了解,今年文總安排的巧思與彩蛋確實相當多,在光雕展演的前段,先以文協百年的主題揭開序幕,中段則是重現民眾幾個月前面臨防疫三級警戒的日常生活,除了空蕩蕩的街道、視訊上班上課、量體溫、酒精消毒、掃實聯制<en>等,還有外送員穿梭街頭送餐的動畫。

<loc0>上演真實版《<unk>》!民眾體驗打起來警鎮壓竄逃畫面曝
<loc1>劇集《<unk>》全球爆紅,繼梨泰院地鐵站增設戲劇體驗的設施供劇迷朝聖後,<loc2>所拍攝的預告片也利用劇中「<num>木頭人娃娃」的恐懼感,禁止人民在過紅綠燈時穿越馬路,創新的宣傳手法吸引目光。近來,<loc0>也在當地建了一間遊戲體驗的快閃咖啡廳,<num>日、<num>日於<loc3>登場,怎料引起暴動,場面混亂、警察出動鎮壓,畫面曝光後,令不少網友驚呼:「他們真的體驗到《<unk>》了!」<loc0><en>片商為宣傳《<unk>》,實際於<loc3>設置出一間體驗咖啡廳,據悉,館內高度還原戲劇裡出現的大型豬存錢筒、劇裡服裝以及戳椪糖關卡,馬上就吸引大批劇迷朝聖,店外車水馬龍。然而,根據當地媒體報導,該咖啡廳人氣太旺,若想進入館內得從前一晚就露宿排隊,正因如此,現場出現了「插隊」動亂,使得一群人在館外打起來,而民眾為了躲避打架紛爭、邊尖叫邊逃難,火爆程度引來警方鎮壓,各種動亂畫面在推特、抖音上流傳,最終結局就是該體驗館提早關閉。

囤房大戶<num>人<org0>加強查緝租金所得漏報
配合<org1><org2>臨時提案,<org0>宣布即日啟動「個人間房屋租賃所得專案查核作業計畫」,首度精準鎖定全國擁有<num>戶以上非自住房屋的<num>名囤房大戶,展開租金所得查核,遏止漏稅。<org0>今天發布新聞稿,並同步於<org3>例行記者會上說明,配合<org1><org4>今年<num>月<num>日通過的臨時提案,將鎖定全國持有<num>戶以上非自住房屋者,逐步清查是否如實申報租金收入,以落實居住正義,避免租金收入成為逃漏稅黑洞。<org3>署長<per0>表示,統計目前全<loc0>持有逾<num>戶以上非自住房屋的個人共<num>名,五區<org5>近日將陸續逐一寄發輔導函,要求前來說明名下房屋是否有出租情形若有出租但未曾就租賃所得繳稅,將輔導補報補繳,此階段僅需補稅仍免罰。

用飲水機洗臉的貓?短腿大眼萌樣紅到國外「原來是美容祕訣!」
大家對貓咪的印象多數是不喜歡弄濕或怕水,<loc0>卻有一隻貓咪「<en>」直接側臉躺在循環飲水機的水裡,一臉舒服的萌樣在推特上爆紅,各國的網友除了大讚<en>太可愛,也疑惑:「都弄濕了沒關係?」飼主則在貼文說<en>是「直接洗臉」的路線,有人就笑稱:「原來是美容祕訣!」現在許多飼主家中會放自動循環流水的飲水機給寵物喝水,<loc0>有位飼主<en>日前在他推特,上傳家中的曼赤肯貓「<en>」使用飲水機的影片,只見牠突然倒下躺著,頭部靠在飲水機邊,側臉臉頰直接躺在流動的水面,還伸出一隻小短腿撥弄流水,後來轉頭換下巴靠在水面,怡然自得地睜著萌萌大眼看向鏡頭。影片上傳推特後爆紅,超過<num>萬次觀看與<num>萬則轉推,不只<loc0>連各國網友都被圈粉留言,「毛都弄濕了欸…」「<en>!」「臉頰不會冷冷的嗎?」「我家也有這台,原來是這樣用的啊。」「貓咪果然是水做的。」而原<en>在貼文上則開玩笑說<en>是「直接洗臉的路線」,因此很多人讚嘆「原來這就是<en>美麗的祕密!」
"""