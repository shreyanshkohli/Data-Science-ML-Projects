from sklearn.model_selection import train_test_split

def optimise(model, x, y):
    best_score = -float('inf')
    best_index = -1
    
    for index, i in enumerate(range(1, 100)):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=i)
        model.fit(xtrain, ytrain)
        score = model.score(xtest, ytest)
        print(score, index)
        
    #     if score > best_score:
    #         best_score = score
    #         best_index = i
    
    # return best_score, best_index

    
feature_columns = [
    "total_sqft", "bath", "balcony", "bedrooms", "location_ devarachikkanahalli", "location_1st block jayanagar", "location_1st block koramangala", "location_1st phase jp nagar", "location_2nd phase judicial layout", "location_2nd stage nagarbhavi", "location_5th block hbr layout", "location_5th phase jp nagar", "location_6th phase jp nagar", "location_7th phase jp nagar", "location_8th phase jp nagar", "location_9th phase jp nagar", "location_aecs layout", "location_abbigere", "location_akshaya nagar", "location_ambalipura", "location_ambedkar nagar", "location_amruthahalli", "location_anandapura", "location_ananth nagar", "location_anekal", "location_anjanapura", "location_ardendale", "location_arekere", "location_attibele", "location_beml layout", "location_btm 1st stage", "location_btm 2nd stage", "location_btm layout", "location_babusapalaya", "location_badavala nagar", "location_balagere", "location_banashankari", "location_banashankari stage ii", "location_banashankari stage iii", "location_banashankari stage v", "location_banashankari stage vi", "location_banaswadi", "location_banjara layout", "location_bannerghatta", "location_bannerghatta road", "location_basapura", "location_basavangudi", "location_basaveshwara nagar", "location_battarahalli", "location_begur", "location_begur road", "location_bellandur", "location_benson town", "location_bharathi nagar", "location_bhoganhalli", "location_billekahalli", "location_binny pete", "location_bisuvanahalli", "location_bommanahalli", "location_bommasandra", "location_bommasandra industrial area", "location_bommenahalli", "location_brookefield", "location_budigere", "location_cv raman nagar", "location_chamrajpet", "location_chandapura", "location_channasandra", "location_chikka tirupathi", "location_chikkabanavar", "location_chikkalasandra", "location_choodasandra", "location_cooke town", "location_cox town", "location_cunningham road", "location_dairy circle", "location_dasanapura", "location_dasarahalli", "location_devanahalli", "location_dodda nekkundi", "location_doddaballapur", "location_doddakallasandra", "location_doddathoguru", "location_dodsworth layout", "location_domlur", "location_dommasandra", "location_epip zone", "location_electronic city", "location_electronic city phase ii", "location_electronics city phase 1", "location_frazer town", "location_gm palaya", "location_ganga nagar", "location_garudachar palya", "location_giri nagar", "location_gollarapalya hosahalli", "location_gottigere", "location_green glen layout", "location_gubbalala", "location_gunjur", "location_gunjur palya", "location_hal 2nd stage", "location_hbr layout", "location_hrbr layout", "location_hsr layout", "location_haralur road", "location_harlur", "location_hebbal", "location_hebbal kempapura", "location_hegde nagar", "location_hennur", "location_hennur road", "location_hoodi", "location_horamavu agara", "location_horamavu banaswadi", "location_hormavu", "location_hosa road", "location_hosakerehalli", "location_hoskote", "location_hosur road", "location_hulimavu", "location_isro layout", "location_itpl", "location_iblur village", "location_indira nagar", "location_jp nagar", "location_jakkur", "location_jalahalli", "location_jalahalli east", "location_jigani", "location_judicial layout", "location_kr puram", "location_kadubeesanahalli", "location_kadugodi", "location_kaggadasapura", "location_kaggalipura", "location_kaikondrahalli", "location_kalena agrahara", "location_kalkere", "location_kalyan nagar", "location_kambipura", "location_kammanahalli", "location_kammasandra", "location_kanakapura", "location_kanakpura road", "location_kannamangala", "location_karuna nagar", "location_kasavanhalli", "location_kasturi nagar", "location_kathriguppe", "location_kaval byrasandra", "location_kenchenahalli", "location_kengeri", "location_kengeri satellite town", "location_kereguddadahalli", "location_kodichikkanahalli", "location_kodigehaali", "location_kodigehalli", "location_kodihalli", "location_kogilu", "location_konanakunte", "location_koramangala", "location_kothannur", "location_kothanur", "location_kudlu", "location_kudlu gate", "location_kumaraswami layout", "location_kundalahalli", "location_lb shastri nagar", "location_laggere", "location_lakshminarayana pura", "location_lingadheeranahalli", "location_magadi road", "location_mahadevpura", "location_mahalakshmi layout", "location_mallasandra", "location_malleshpalya", "location_malleshwaram", "location_marathahalli", "location_margondanahalli", "location_marsur", "location_mico layout", "location_munnekollal", "location_murugeshpalya", "location_mysore road", "location_ngr layout", "location_nri layout", "location_nagadevanahalli", "location_naganathapura", "location_nagappa reddy layout", "location_nagarbhavi", "location_nagasandra", "location_nagavara", "location_nagavarapalya", "location_narayanapura", "location_neeladri nagar", "location_nehru nagar", "location_ombr layout", "location_old airport road", "location_old madras road", "location_padmanabhanagar", "location_pai layout", "location_panathur", "location_parappana agrahara", "location_pattandur agrahara", "location_poorna pragna layout", "location_prithvi layout", "location_r.t. nagar", "location_rachenahalli", "location_raja rajeshwari nagar", "location_rajaji nagar", "location_rajiv nagar", "location_ramagondanahalli", "location_ramamurthy nagar", "location_rayasandra", "location_sadashiva nagar", "location_sahakara nagar", "location_sanjay nagar", "location_sarakki nagar", "location_sarjapur", "location_sarjapur  road", "location_sarjapura - attibele road", "location_sector 2 hsr layout", "location_sector 7 hsr layout", "location_seegehalli", "location_shampura", "location_shivaji nagar", "location_singasandra", "location_somasundara palya", "location_sompura", "location_sonnenahalli", "location_subramanyapura", "location_sultan palaya", "location_tc palaya", "location_talaghattapura", "location_thanisandra", "location_thigalarapalya", "location_thubarahalli", "location_thyagaraja nagar", "location_tindlu", "location_tumkur road", "location_ulsoor", "location_uttarahalli", "location_varthur", "location_varthur road", "location_vasanthapura", "location_vidyaranyapura", "location_vijayanagar", "location_vishveshwarya layout", "location_vishwapriya layout", "location_vittasandra", "location_whitefield", "location_yelachenahalli", "location_yelahanka", "location_yelahanka new town", "location_yelenahalli", "location_yeshwanthpur"
]